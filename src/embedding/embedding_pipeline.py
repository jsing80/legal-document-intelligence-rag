
import hashlib
import chromadb

from sentence_transformers import SentenceTransformer
from config.settings import (
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_READ_BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
    MODEL_INFERENCE_BATCH_SIZE  
)
from ingestion.chunk_persistence import load_chunks
from utils.logger import get_logger


logger = get_logger(__name__)

def calculate_content_hash(text):
    """Generate a hash using chunk text and embedding configuration."""

    # Include model configuration so changes trigger re-embedding.
    content = f"{EMBEDDING_MODEL}|normalize_embeddings=True|{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def run_embedding_pipeline():
    """Compare source chunks with ChromaDB and incrementally store embeddings."""

    # --------------------------------------------------
    # TASK 1: Load source chunks
    # --------------------------------------------------
    source_chunks = load_chunks()

    if not source_chunks:
        logger.warning("No source chunks found. Pipeline stopped.")
        return

    logger.info("Loaded %d source chunks", len(source_chunks))

    # Collect all current source IDs for the later deletion check.
    source_chunk_ids = {
        chunk["metadata"]["chunk_id"] for chunk in source_chunks
    }

    # --------------------------------------------------
    # TASK 2: Initialise persistent ChromaDB
    # --------------------------------------------------
    # PersistentClient stores vectors on disk and reopens the database on reruns.
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("Existing vectors in ChromaDB: %d", collection.count())

    # --------------------------------------------------
    # TASK 3: Read existing chunk IDs and hashes
    # --------------------------------------------------
    existing_chunk_hashes = {}
    offset = 0

    while True:
        # Read IDs and metadata only; existing embedding vectors are not needed.
        result = collection.get(
            limit=CHROMA_READ_BATCH_SIZE,
            offset=offset,
            include=["metadatas"]
        )

        if not result["ids"]:
            break

        # Store each existing ID with its hash for quick comparison.
        for existing_chunk_id, metadata in zip(result["ids"], result["metadatas"]):
            existing_chunk_hashes[existing_chunk_id] = (
                metadata.get("content_hash") if metadata else None
            )

        offset += len(result["ids"])

    existing_chunk_ids = set(existing_chunk_hashes)

    logger.info("Retrieved %d existing chunk records", len(existing_chunk_ids))

    # --------------------------------------------------
    # TASK 4: Load embedding model
    # --------------------------------------------------
    # Load the model once and reuse it for all source batches.
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Track pipeline results.
    new_count = 0
    existing_modified_count = 0
    existing_unchanged_count = 0
    upserted_count = 0

    # --------------------------------------------------
    # TASK 5: Process source chunks batch by batch
    # --------------------------------------------------
    for batch_start in range(0, len(source_chunks), EMBEDDING_BATCH_SIZE):

        # Select the next batch of source chunks.
        source_batch = source_chunks[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

        # These lists contain only new or modified chunks in this batch.
        new_or_modified_ids = []
        new_or_modified_texts = []
        new_or_modified_metadatas = []

        for chunk in source_batch:
            chunk_id = chunk["metadata"]["chunk_id"]
            chunk_text = chunk["text"]

            # Calculate the current hash for comparison with ChromaDB.
            current_hash = calculate_content_hash(chunk_text)

            if chunk_id in existing_chunk_ids:

                if existing_chunk_hashes[chunk_id] == current_hash:
                    # Existing ID and matching hash: reuse the stored embedding.
                    existing_unchanged_count += 1
                    continue

                # Existing ID but different hash: regenerate its embedding.
                existing_modified_count += 1

            else:
                # New ID: generate an embedding for the new chunk.
                new_count += 1

            # Attach the current hash so future runs can detect changes.
            metadata = {
                **chunk["metadata"],
                "content_hash": current_hash,
                "embedding_model": EMBEDDING_MODEL
            }

            new_or_modified_ids.append(chunk_id)
            new_or_modified_texts.append(chunk_text)
            new_or_modified_metadatas.append(metadata)

        # --------------------------------------------------
        # TASK 6: Generate embeddings and upsert this batch
        # --------------------------------------------------
        if new_or_modified_ids:

            # Generate normalized vectors only for new or modified chunks.
            embeddings = model.encode(
                new_or_modified_texts,
                batch_size=MODEL_INFERENCE_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Insert new IDs or replace existing records with updated embeddings.
            collection.upsert(
                ids=new_or_modified_ids,
                embeddings=embeddings.tolist(),
                documents=new_or_modified_texts,
                metadatas=new_or_modified_metadatas
            )

            upserted_count += len(new_or_modified_ids)

        logger.info(
            "Processed source batch %d | New/modified embedded: %d",
            batch_start // EMBEDDING_BATCH_SIZE + 1,
            len(new_or_modified_ids)
        )

    # --------------------------------------------------
    # TASK 7: Identify and delete obsolete chunks
    # --------------------------------------------------
    # Existing ChromaDB IDs absent from the current source are obsolete.
    deleted_ids = list(existing_chunk_ids - source_chunk_ids)

    # Delete only after every source batch has been processed successfully.
    if deleted_ids:
        collection.delete(ids=deleted_ids)

    # --------------------------------------------------
    # TASK 8: Log final pipeline summary
    # --------------------------------------------------
    logger.info(
        "Pipeline completed | New: %d | Existing modified: %d | "
        "Existing unchanged: %d | Upserted: %d | Deleted: %d",
        new_count,
        existing_modified_count,
        existing_unchanged_count,
        upserted_count,
        len(deleted_ids)
    )

    logger.info("Total vectors now stored: %d", collection.count())


if __name__ == "__main__":
    run_embedding_pipeline()
