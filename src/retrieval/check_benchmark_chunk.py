from ingestion.chunk_persistence import load_chunks

def main():
    """Find chunks overlapping the benchmark's gold evidence span."""
    gold_path = "contractnli/CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt"
    gold_start, gold_end = 11461, 11963

    chunks = load_chunks()
    matching_chunks = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        source_path = metadata["source_path"].replace("\\", "/")

        # Match the benchmark document using its relative path suffix.
        if not source_path.endswith(gold_path):
            continue

        chunk_start = metadata["start_char"]
        chunk_end = metadata["end_char"]

        # Check whether the chunk overlaps the benchmark evidence span.
        overlap = max(0, min(chunk_end, gold_end) - max(chunk_start, gold_start))

        if overlap > 0:
            matching_chunks.append((chunk, overlap))

    print(f"Matching evidence chunks: {len(matching_chunks)}")

    for chunk, overlap in matching_chunks:
        metadata = chunk["metadata"]
        print(f"\nChunk ID: {metadata['chunk_id']}")
        print(f"Chunk span: {metadata['start_char']}–{metadata['end_char']}")
        print(f"Gold span overlap: {overlap} characters")
        print(f"Text: {chunk['text']}")

if __name__ == "__main__":
    main()