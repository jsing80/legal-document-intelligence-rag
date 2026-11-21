
import json
import shutil
from collections import Counter
from pathlib import Path

import chromadb

from config.settings import (
    CORPUS_DIR,
    CHUNKS_PATH,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    BM25_INDEX_DIR,
    BM25_CORPUS_PATH
)

from ingestion.document_loader import load_documents
from ingestion.document_persistence import save_documents
from ingestion.chunking import (
    chunk_documents,
    get_content_hash,
    get_chunking_config
)
from ingestion.chunk_persistence import save_chunks
from ingestion.chunk_manifest import save_manifest

from embedding.embedding_pipeline import run_embedding_pipeline
from ingestion.bm25_index import build_bm25_index


def is_inside_corpus(source_path):
    """Check whether a stored source path belongs to the configured corpus."""

    if not source_path:
        return False

    try:
        return Path(source_path).resolve().is_relative_to(
            CORPUS_DIR.resolve()
        )
    except (ValueError, OSError):
        return False


def rebuild_documents_and_chunks():
    """Reload the current corpus and regenerate all base chunks."""

    print("\nSTEP 1: Loading current corpus")

    documents = load_documents(CORPUS_DIR)

    if not documents:
        raise ValueError(
            f"No documents found in configured corpus: {CORPUS_DIR}"
        )

    print(f"Documents loaded: {len(documents)}")

    save_documents(documents)

    print("\nSTEP 2: Recreating chunks from scratch")

    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("Chunk generation returned no chunks.")

    save_chunks(chunks)

    chunk_counts = Counter(
        chunk["metadata"]["document_id"]
        for chunk in chunks
    )

    manifest_documents = {}

    for document in documents:
        document_id = document["document_id"]

        manifest_documents[document_id] = {
            "content_hash": get_content_hash(
                document["processed_text"]
            ),
            "chunk_count": chunk_counts[document_id]
        }

    save_manifest({
        "chunking_config": get_chunking_config(),
        "documents": manifest_documents
    })

    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunked documents: {len(chunk_counts)}")

    invalid_paths = [
        chunk["metadata"].get("source_path", "")
        for chunk in chunks
        if not is_inside_corpus(
            chunk["metadata"].get("source_path", "")
        )
    ]

    if invalid_paths:
        raise ValueError(
            "Old or invalid source paths found in regenerated chunks:\n"
            + "\n".join(invalid_paths[:10])
        )

    print("Chunk source-path validation: PASSED")

    return chunks


def rebuild_chroma():
    """Delete the existing collection and recreate embeddings from current chunks."""

    print("\nSTEP 3: Rebuilding ChromaDB")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    # Remove only this project's configured collection.
    client.delete_collection(
        name=CHROMA_COLLECTION_NAME
    )

    print("Existing Chroma collection deleted.")

    # Recreate the collection and embed all regenerated chunks.
    run_embedding_pipeline()

    collection = client.get_collection(
        name=CHROMA_COLLECTION_NAME
    )

    print(f"Chroma records: {collection.count()}")

    invalid_paths = []
    offset = 0
    batch_size = 5000

    while True:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )

        ids = result["ids"]
        metadatas = result["metadatas"] or []

        for metadata in metadatas:
            source_path = metadata.get("source_path", "")

            if not is_inside_corpus(source_path):
                invalid_paths.append(source_path)

        if len(ids) < batch_size:
            break

        offset += len(ids)

    if invalid_paths:
        raise ValueError(
            "Old or invalid source paths found in ChromaDB:\n"
            + "\n".join(invalid_paths[:10])
        )

    print("Chroma source-path validation: PASSED")


def rebuild_bm25():
    """Remove the old BM25 index and rebuild it from regenerated chunks."""

    print("\nSTEP 4: Rebuilding BM25")

    # Remove both the persisted index and its parallel corpus metadata.
    if BM25_INDEX_DIR.exists():
        shutil.rmtree(BM25_INDEX_DIR)

    # Build a fresh index from CHUNKS_PATH.
    build_bm25_index()

    if not BM25_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"BM25 corpus not created: {BM25_CORPUS_PATH}"
        )

    corpus = []

    with BM25_CORPUS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            if line.strip():
                corpus.append(json.loads(line))

    invalid_paths = [
        item["metadata"].get("source_path", "")
        for item in corpus
        if not is_inside_corpus(
            item["metadata"].get("source_path", "")
        )
    ]

    if invalid_paths:
        raise ValueError(
            "Old or invalid source paths found in BM25 corpus:\n"
            + "\n".join(invalid_paths[:10])
        )

    print(f"BM25 indexed chunks: {len(corpus)}")
    print("BM25 source-path validation: PASSED")


def main():
    print("=" * 65)
    print("LEGAL RAG - CLEAN RETRIEVAL INDEX REBUILD")
    print("=" * 65)

    print(f"\nConfigured corpus: {CORPUS_DIR}")

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(
            f"Configured corpus does not exist: {CORPUS_DIR}"
        )

    rebuild_documents_and_chunks()
    rebuild_chroma()
    rebuild_bm25()

    print("\n" + "=" * 65)
    print("REBUILD COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"Chunks: {CHUNKS_PATH}")
    print(f"Chroma: {CHROMA_DB_DIR}")
    print(f"BM25 index: {BM25_INDEX_DIR}")
    print(f"BM25 corpus: {BM25_CORPUS_PATH}")


if __name__ == "__main__":
    main()

