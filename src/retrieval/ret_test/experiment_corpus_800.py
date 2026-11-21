
import json
import math
import chromadb
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config.settings import (
    PROJECT_ROOT,
    PROCESSED_DOCUMENTS_PATH,
    CHROMA_DB_DIR,
    QUERY_INSTRUCTION,
    EMBEDDING_BATCH_SIZE,
    MODEL_INFERENCE_BATCH_SIZE
)

# Full-corpus experiment configuration.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "legal_800_bge_base"

CHUNKS_DIR = PROJECT_ROOT / "chunks_experiments" / "corpus_800_100"
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"

QUERY = (
    "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; "
    "Does the document indicate that the Agreement does not grant the "
    "Receiving Party any rights to the Confidential Information?"
)

GOLD_SOURCE_PATH = "contractnli/CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt"
GOLD_START, GOLD_END = 11461, 11963


def generate_all_chunks():
    """Generate and persist chunks from every preprocessed document."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ""],
        add_start_index=True
    )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    document_count = 0
    chunk_count = 0

    with PROCESSED_DOCUMENTS_PATH.open("r", encoding="utf-8") as source_file, \
         CHUNKS_PATH.open("w", encoding="utf-8") as output_file:

        for line in source_file:
            if not line.strip():
                continue

            document = json.loads(line)
            document_count += 1

            # Split each document independently to preserve its source offsets.
            split_documents = splitter.create_documents(
                [document["processed_text"]]
            )

            for index, item in enumerate(split_documents):
                chunk_text = item.page_content
                start_char = item.metadata["start_index"]

                metadata = {
                    "chunk_id": (
                        f"{document['document_id']}::experiment_800_{index:04d}"
                    ),
                    "document_id": str(document["document_id"]),
                    "source_file": str(document["source_file"]),
                    "source_path": str(document["source_path"]).replace("\\", "/"),
                    "chunk_index": index,
                    "start_char": start_char,
                    "end_char": start_char + len(chunk_text)
                }

                # Persist original legal text separately from its metadata.
                output_file.write(json.dumps({
                    "metadata": metadata,
                    "text": chunk_text
                }, ensure_ascii=False) + "\n")

                chunk_count += 1

            if document_count % 100 == 0:
                print(f"Processed {document_count} documents; {chunk_count} chunks")

    print(f"\nDocuments processed: {document_count}")
    print(f"Total chunks generated: {chunk_count}")
    print(f"Chunk file: {CHUNKS_PATH}")

    return chunk_count


def read_chunk_batches(batch_size):
    """Read persisted chunks in batches to control memory usage."""
    batch = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            batch.append(json.loads(line))

            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def create_vector_collection(chunk_count):
    """Embed all chunks and store them in a dedicated ChromaDB collection."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Recreate only this experiment's collection to prevent stale records.
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }

    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    total_batches = math.ceil(chunk_count / EMBEDDING_BATCH_SIZE)

    for batch in tqdm(
        read_chunk_batches(EMBEDDING_BATCH_SIZE),
        total=total_batches,
        desc="Embedding and indexing"
    ):
        texts = [item["text"] for item in batch]
        embeddings = model.encode(
            texts,
            batch_size=MODEL_INFERENCE_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Upsert each batch rather than holding the full corpus in memory.
        collection.upsert(
            ids=[item["metadata"]["chunk_id"] for item in batch],
            documents=texts,
            metadatas=[item["metadata"] for item in batch],
            embeddings=embeddings.tolist()
        )

    print(f"\nChromaDB collection: {COLLECTION_NAME}")
    print(f"Indexed records: {collection.count()}")

    return collection, model


def normalize_path(path):
    """Normalize source paths for reliable benchmark matching."""
    return str(path).replace("\\", "/").lstrip("./").casefold()


def run_benchmark_query(collection, model):
    """Retrieve corpus-wide top results and measure gold evidence coverage."""
    query_embedding = model.encode(
        [QUERY_INSTRUCTION + QUERY],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(20, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    print("\nCORPUS-WIDE TOP 20 RESULTS\n")
    gold_matches = []

    for rank, (text, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), start=1):

        similarity = 1.0 - distance
        source_path = normalize_path(metadata["source_path"])

        print(
            f"\nRank: {rank} | Similarity: {similarity:.4f}\n"
            f"Source: {source_path}\n"
            f"Chunk: {metadata['chunk_id']}\n"
            f"Span: {metadata['start_char']}-{metadata['end_char']}"
        )
        print(text[:500])

        # Match both document identity and character-span overlap.
        is_gold_document = (
            source_path == normalize_path(GOLD_SOURCE_PATH)
            or source_path.endswith("/" + normalize_path(GOLD_SOURCE_PATH))
        )

        if is_gold_document:
            overlap = max(
                0,
                min(metadata["end_char"], GOLD_END)
                - max(metadata["start_char"], GOLD_START)
            )

            if overlap > 0:
                gold_matches.append({
                    "rank": rank,
                    "chunk_id": metadata["chunk_id"],
                    "overlap_chars": overlap,
                    "coverage_percent": round(
                        overlap / (GOLD_END - GOLD_START) * 100, 2
                    )
                })

    print("\nGOLD EVIDENCE MATCHES")
    print(json.dumps(gold_matches, indent=2))
    print(
        "\nGold evidence found in corpus-wide top 20:",
        bool(gold_matches)
    )


if __name__ == "__main__":
    total_chunks = generate_all_chunks()
    collection, model = create_vector_collection(total_chunks)
    run_benchmark_query(collection, model)
