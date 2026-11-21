
from retrieval.retrieval_pipeline import LegalRetrievalPipeline
from generation.generator import generate_answer


def main():
    """Run retrieval and generate an answer using the retrieved legal evidence."""

    # Initialize models and retrieval resources once.
    pipeline = LegalRetrievalPipeline()

    user_query = "What restrictions does the agreement impose on Mentor regarding disclosure of Confidential Information?"

    if not user_query.strip():
        print("Please enter a valid query.")
        return

    # Run dense retrieval, BM25, RRF fusion and cross-encoder reranking.
    results = pipeline.retrieve(user_query)

    # Inspect the results produced at each retrieval stage.
    print(f"\nQuery: {user_query}")

    for stage in ["dense_results", "bm25_results", "fused_results", "final_results"]:
        print(f"\n{'=' * 60}\n{stage.upper()}\n{'=' * 60}")

        for rank, chunk in enumerate(results[stage], start=1):
            score = chunk.get(
                "reranker_score",
                chunk.get("rrf_score", chunk.get("dense_score", chunk.get("bm25_score")))
            )

            print(f"\nRank: {rank}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Score: {score}")
            print(f"Text: {chunk['text']}")

    # Pass only the final reranked chunks to Gemini for answer generation.
    final_results = results["final_results"]
    answer = generate_answer(user_query, final_results)

    print(f"\n{'=' * 60}\nGENERATED ANSWER\n{'=' * 60}")
    print(answer)


if __name__ == "__main__":
    main()
