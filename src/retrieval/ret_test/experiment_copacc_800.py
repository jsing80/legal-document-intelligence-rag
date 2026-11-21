
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config.settings import (
    PROCESSED_DOCUMENTS_PATH,
    EXPERIMENT_CHUNK_SIZE,
    EXPERIMENT_CHUNK_OVERLAP,
    EXPERIMENT_EMBEDDING_MODEL,
    EXPERIMENT_CHUNKS_DIR,
    EXPERIMENT_CHUNKS_PATH,
    CHROMA_DB_DIR,
    EXPERIMENT_CHROMA_COLLECTION,
    QUERY_INSTRUCTION
)

# Benchmark document and question for this experiment.
TARGET_FILE = "CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt"
QUERY = (
    "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; "
    "Does the document indicate that the Agreement does not grant the "
    "Receiving Party any rights to the Confidential Information?"
)

def load_target_document():
    """Load only the CopAcc document from the processed document store."""
    with PROCESSED_DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            document = json.loads(line)
            if document["source_file"] == TARGET_FILE:
                return document
    raise ValueError(f"Document not found: {TARGET_FILE}")

def create_experiment_chunks(document):
    """Create 800-character chunks while retaining source offsets."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=EXPERIMENT_CHUNK_SIZE,
        chunk_overlap=EXPERIMENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ""],
        add_start_index=True
    )

    split_documents = splitter.create_documents([document["processed_text"]])
    chunks = []

    for index, item in enumerate(split_documents):
        chunk_text = item.page_content
        start_char = item.metadata["start_index"]

        # Preserve the original document identity and character offsets.
        metadata = {
            "chunk_id": f"{document['document_id']}::experiment_800_{index:04d}",
            "document_id": document["document_id"],
            "source_file": document["source_file"],
            "source_path": document["source_path"],
            "chunk_index": index,
            "start_char": start_char,
            "end_char": start_char + len(chunk_text)
        }

        chunks.append({"metadata": metadata, "text": chunk_text})

    return chunks

def save_experiment_chunks(chunks):
    """Persist experiment chunks separately from the V1 baseline."""
    EXPERIMENT_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_CHUNKS_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

def embed_and_index(chunks):
    """Generate normalized BGE-large embeddings and store them in a new collection."""
    model = SentenceTransformer(EXPERIMENT_EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Create a fresh collection; do not reuse or overwrite the baseline.
    collection = client.get_or_create_collection(
        name=EXPERIMENT_CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Encode the actual chunk text, using the same normalization approach as V1.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    collection.upsert(
        ids=[chunk["metadata"]["chunk_id"] for chunk in chunks],
        documents=texts,
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings.tolist()
    )

    print(f"Indexed {collection.count()} chunks in {EXPERIMENT_CHROMA_COLLECTION}")
    return collection, model

def run_query(collection, model):
    """Retrieve the top 20 chunks for the original benchmark question."""
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

    print("\nTOP RETRIEVED CHUNKS\n")

    for rank, (text, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), start=1):
        print(
            f"\nRank: {rank} | Similarity: {1 - distance:.4f} | "
            f"Chunk: {metadata['chunk_id']} | "
            f"Span: {metadata['start_char']}-{metadata['end_char']}"
        )
        print(text)

    # Check whether a retrieved chunk intersects the benchmark evidence span.
    gold_start, gold_end = 11461, 11963
    gold_ranks = [
        rank for rank, metadata in enumerate(results["metadatas"][0], start=1)
        if metadata["start_char"] < gold_end
        and metadata["end_char"] > gold_start
    ]
    print(f"\nGold evidence chunk rank(s): {gold_ranks or 'Not in top 20'}")

if __name__ == "__main__":
    document = load_target_document()
    chunks = create_experiment_chunks(document)
    save_experiment_chunks(chunks)

    print(f"Source document: {document['source_file']}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Chunk size: {EXPERIMENT_CHUNK_SIZE}; overlap: {EXPERIMENT_CHUNK_OVERLAP}")

    collection, model = embed_and_index(chunks)
    run_query(collection, model)
