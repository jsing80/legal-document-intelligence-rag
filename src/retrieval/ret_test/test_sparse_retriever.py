from retrieval.sparse_retriever import BM25Retriever

def main():
    """Verify BM25 index loading and retrieval of original legal chunks. This is only for testing purpose created."""
    
    retriever = BM25Retriever()
    top_k=20
    query = (
    "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; "
    "Does the document indicate that the Agreement does not grant the "
    "Receiving Party any rights to the Confidential Information?"
    )
    gold_chunks = {"chunk_0035", "chunk_0036"}

    results = retriever.retrieve(query, top_k=top_k)

    print(f"\nQuery: {query}")
    print(f"Results returned: {len(results)}")

    # Check whether the benchmark evidence chunks appear in the ranking.
    for target in gold_chunks:
        matching_ranks = [
            rank for rank, chunk in enumerate(results, start=1)
            if chunk["chunk_id"].endswith(f"::{target}")
        ]
        print(f"{target}: rank {matching_ranks[0]}" if matching_ranks else f"{target}: NOT FOUND in top {top_k}")

    # Display the retrieved candidates for inspection.
    for rank, chunk in enumerate(results, start=1):
        score = chunk.get("dense_score", chunk.get("bm25_score"))
        print(f"\nRank: {rank} | Score: {score:.4f}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Text: {chunk['text'][:300]}")

if __name__ == "__main__":
    main()