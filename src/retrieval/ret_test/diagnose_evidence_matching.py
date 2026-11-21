from ingestion.chunk_persistence import load_chunks
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import BM25Retriever

def main():
    """Compare natural-language and exact-evidence retrieval for gold chunks."""

    target_ids = {
        r"contractnli\CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt::chunk_0035",
        r"contractnli\CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt::chunk_0036"
    }

    # Load the authoritative source chunks.
    chunks = load_chunks()
    targets = {
        chunk["metadata"]["chunk_id"]: chunk
        for chunk in chunks
        if chunk["metadata"]["chunk_id"] in target_ids
    }

    dense = DenseRetriever()
    sparse = BM25Retriever()

    # Check that stored index content matches the source chunk content.
    print("\nINDEX CONTENT VALIDATION")

    dense_data = dense.collection.get(
        ids=list(target_ids),
        include=["documents", "metadatas"]
    )
    dense_documents = dict(zip(dense_data["ids"], dense_data["documents"]))

    sparse_documents = {
        item["chunk_id"]: item["text"]
        for item in sparse.corpus
    }

    for chunk_id, source_chunk in targets.items():
        source_text = source_chunk["text"]
        print(f"\n{chunk_id}")
        print(f"Chroma text matches source: {dense_documents.get(chunk_id) == source_text}")
        print(f"BM25 text matches source: {sparse_documents.get(chunk_id) == source_text}")

    # Compare the original question with exact evidence wording.
    natural_query = (
        "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; "
        "Does the document indicate that the Agreement does not grant the "
        "Receiving Party any rights to the Confidential Information?"
    )

    evidence_query = targets[next(iter(targets))]["text"]

    for query_name, query in [
        ("NATURAL LANGUAGE QUESTION", natural_query),
        ("EXACT EVIDENCE TEXT", evidence_query)
    ]:
        print(f"\n{'=' * 65}\n{query_name}\n{'=' * 65}")

        for name, retriever in [("DENSE", dense), ("BM25", sparse)]:
            results = retriever.retrieve(query, top_k=100)
            ranks = {
                item["chunk_id"]: rank
                for rank, item in enumerate(results, start=1)
            }

            print(f"\n{name}")
            for chunk_id in target_ids:
                print(
                    f"{chunk_id}: rank {ranks[chunk_id]}"
                    if chunk_id in ranks
                    else f"{chunk_id}: NOT IN TOP 100"
                )

if __name__ == "__main__":
    main()