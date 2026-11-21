import json
from config.settings import CHUNKS_PATH, EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_token_lengths():
    """Measure token lengths across persisted chunks before embedding."""

    # Load BGE and access its tokenizer to measure actual model tokens.
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer
    # Read the model's configured maximum input sequence length.
    max_seq_length = model.max_seq_length

    token_lengths = []
    oversized_chunks = []

    # Process the JSONL file line by line to limit memory usage.
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)
            text = chunk["text"]

            # Count tokens without truncation to identify oversized chunks.
            token_ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )["input_ids"]

            token_count = len(token_ids)
            token_lengths.append(token_count)

            # Record chunks that exceed BGE's input limit.
            if token_count > max_seq_length:
                oversized_chunks.append({
                    "chunk_id": chunk["metadata"]["chunk_id"],
                    "token_count": token_count
                })

    # Sort token counts to calculate distribution statistics.
    token_lengths.sort()
    total_chunks = len(token_lengths)

    if not total_chunks:
        print("No chunks found.")
        return

    # Display token-length statistics for the entire corpus.
    print("\nEmbedding input validation")
    print("Embedding model:", EMBEDDING_MODEL)
    print("Model maximum sequence length:", max_seq_length)
    print("Total chunks:", total_chunks)
    print("Minimum tokens:", token_lengths[0])
    print("Average tokens:", round(sum(token_lengths) / total_chunks, 2))
    print("95th percentile:", token_lengths[int(total_chunks * 0.95) - 1])
    print("99th percentile:", token_lengths[int(total_chunks * 0.99) - 1])
    print("Maximum tokens:", token_lengths[-1])
    print("Chunks exceeding model limit:", len(oversized_chunks))

    # Print examples of oversized chunks for further inspection.
    for chunk in oversized_chunks[:20]:
        print(chunk)


if __name__ == "__main__":
    # Run validation when this file is executed directly.
    validate_token_lengths()