import json

from config.settings import CHUNKS_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

def save_chunks(chunks, output_path=CHUNKS_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Persisted %d chunks to %s", len(chunks), output_path)

def load_chunks(input_path=CHUNKS_PATH):
    """Load previously generated chunks, or return an empty list on first run."""

    # No chunk file means this is the first run; there are no existing chunks.
    if not input_path.exists():
        logger.info("No existing chunk file found. Starting fresh.")
        return []

    with input_path.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    logger.info("Loaded %d existing chunks", len(chunks))
    return chunks


if __name__ == "__main__":
    # Load the persisted chunks for validation.
    chunks = load_chunks()

    if not chunks:
        print("No chunks found.")
    else:
        # Extract chunk IDs and text lengths for validation.
        chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
        sizes = [len(chunk["text"]) for chunk in chunks]

        print("\nTotal chunks:", len(chunks))
        print("Unique chunk IDs:", len(set(chunk_ids)))
        print("Minimum characters:", min(sizes))
        print("Average characters:", round(sum(sizes) / len(sizes), 2))
        print("Maximum characters:", max(sizes))
        print("Missing text:", sum(not chunk["text"].strip() for chunk in chunks))
        print("Missing document IDs:", sum(not chunk["metadata"].get("document_id") for chunk in chunks))

        # Identify unusually short chunks for inspection.
        short_chunks = [chunk for chunk in chunks if len(chunk["text"].strip()) < 50]
        print("\nChunks below 50 characters:", len(short_chunks))

        # Display a sample to inspect short chunks.
        for chunk in short_chunks[:20]:
            print(
                "\nChunk ID:", chunk["metadata"]["chunk_id"],
                "\nLength:", len(chunk["text"]),
                "\nText:", repr(chunk["text"])
            )

        # Display one complete chunk for inspection.
        print("\nSample chunk:")
        print(chunks[0])