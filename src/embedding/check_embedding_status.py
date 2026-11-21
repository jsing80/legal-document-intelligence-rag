import chromadb

from config.settings import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_READ_BATCH_SIZE
)
from ingestion.chunk_persistence import load_chunks
from embedding.embedding_pipeline import calculate_content_hash


def check_embedding_status():
    """Compare source chunks against persisted ChromaDB records without generating embeddings."""

    # Load source chunks and calculate their expected hashes.
    source_chunks = load_chunks()
    source_hashes = {
        chunk["metadata"]["chunk_id"]: calculate_content_hash(chunk["text"])
        for chunk in source_chunks
    }

    # Open the existing database; do not create a new collection.
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    # Read stored IDs and hashes in batches; vector data is not retrieved.
    stored_hashes = {}
    offset = 0

    while True:
        result = collection.get(
            limit=CHROMA_READ_BATCH_SIZE,
            offset=offset,
            include=["metadatas"]
        )

        ids = result["ids"]
        metadatas = result["metadatas"] or []

        for chunk_id, metadata in zip(ids, metadatas):
            stored_hashes[chunk_id] = (
                metadata.get("content_hash") if metadata else None
            )

        if len(ids) < CHROMA_READ_BATCH_SIZE:
            break

        offset += len(ids)

    # Classify source chunks by whether their stored embeddings are current.
    matching = sum(
        1 for chunk_id, expected_hash in source_hashes.items()
        if stored_hashes.get(chunk_id) == expected_hash
    )
    missing = sum(
        1 for chunk_id in source_hashes
        if chunk_id not in stored_hashes
    )
    modified = sum(
        1 for chunk_id, expected_hash in source_hashes.items()
        if chunk_id in stored_hashes
        and stored_hashes[chunk_id] != expected_hash
    )
    obsolete = len(set(stored_hashes) - set(source_hashes))

    print("\n========== EMBEDDING STATUS ==========")
    print(f"Source chunks:             {len(source_hashes):,}")
    print(f"ChromaDB records:           {collection.count():,}")
    print(f"Successfully embedded:      {matching:,}")
    print(f"Missing embeddings:         {missing:,}")
    print(f"Changed / needs re-embed:   {modified:,}")
    print(f"Obsolete stored records:    {obsolete:,}")
    print(f"Remaining to embed:         {missing + modified:,}")
    print("======================================")

if __name__ == "__main__":
    check_embedding_status()