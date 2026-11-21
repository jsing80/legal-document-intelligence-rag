from ingestion.chunk_persistence import load_chunks
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import BM25Retriever

def main():
    """Check benchmark chunk presence and ranking across both retrieval indexes."""

    target_path = "CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt"
    target_indices = {35, 36}

    # Find the exact benchmark chunk IDs in the source chunk file.
    chunks = load_chunks()
    targets = [
        chunk for chunk in chunks
        if chunk["metadata"]["source_file"] == target_path
        and chunk["metadata"]["chunk_index"] in target_indices
    ]

    target_ids = [chunk["metadata"]["chunk_id"] for chunk in targets]

    print("\nBENCHMARK CHUNKS")
    for chunk in targets:
        print(chunk["metadata"]["chunk_id"])
    print(f"Total target chunks: {len(target_ids)}")

    # Verify that both indexes contain these exact chunk IDs.
    dense = DenseRetriever()
    dense_stored = dense.collection.get(
        ids=target_ids,
        include=["documents", "metadatas"]
    )

    sparse = BM25Retriever()
    sparse_positions = {
        item["chunk_id"]: position
        for position, item in enumerate(sparse.corpus)
    }

    print("\nINDEX PRESENCE")
    print(f"ChromaDB: {len(dense_stored['ids'])}/{len(target_ids)}")
    print(f"BM25 corpus: {sum(chunk_id in sparse_positions for chunk_id in target_ids)}/{len(target_ids)}")

    # Check whether the target chunks appear deeper in either ranking.
    query = (
        "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; "
        "Does the document indicate that the Agreement does not grant the "
        "Receiving Party any rights to the Confidential Information?"
    )

    for name, retriever in [("DENSE", dense), ("BM25", sparse)]:
        results = retriever.retrieve(query, top_k=100)
        ranks = {
            chunk["chunk_id"]: rank
            for rank, chunk in enumerate(results, start=1)
        }

        print(f"\n{name} RANKING (top 100)")
        for chunk_id in target_ids:
            print(f"{chunk_id}: rank {ranks[chunk_id]}" if chunk_id in ranks else f"{chunk_id}: NOT FOUND")

if __name__ == "__main__":
    main()