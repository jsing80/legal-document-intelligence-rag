
from config.settings import (
    DENSE_RETRIEVAL_K,
    BM25_RETRIEVAL_K,
    FINAL_CONTEXT_K
)
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import LegalReranker


class LegalRetrievalPipeline:
    """Orchestrate dense retrieval, BM25, RRF and reranking."""

    def __init__(self):
        # Initialize each retriever once, not once per user query.
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = BM25Retriever()
        self.reranker = LegalReranker()

    def retrieve(self, query):
        """Return the final ranked legal chunks for a user query."""

        # Stage 1: Retrieve semantically relevant chunks.
        dense_results = self.dense_retriever.retrieve(query,top_k=DENSE_RETRIEVAL_K)

        # Stage 2: Retrieve keyword-relevant chunks using BM25.
        bm25_results = self.sparse_retriever.retrieve(query,top_k=BM25_RETRIEVAL_K)

        # Stage 3: Merge dense and BM25 rankings.
        fused_results = reciprocal_rank_fusion(dense_results,bm25_results,
                                               top_k=DENSE_RETRIEVAL_K + BM25_RETRIEVAL_K)
       
        # Stage 4: Rerank the fused candidates using the cross-encoder.
        final_results = self.reranker.rerank(query,fused_results,top_n=FINAL_CONTEXT_K)

       
        return {
            "query": query,
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "fused_results": fused_results,
            "final_results": final_results
        }
    


if __name__ == "__main__":
    # Run one sample query to inspect the complete retrieval pipeline.
    pipeline = LegalRetrievalPipeline()

    query = (
        "Does the agreement grant the receiving party any rights "
        "to confidential information?"
    )

    results = pipeline.retrieve(query)

    print("\n========== FINAL RETRIEVAL RESULTS ==========")

    for rank, chunk in enumerate(results["final_results"], start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['metadata'].get('source_file')}")
        print(f"RRF score: {chunk['rrf_score']:.5f}")
        print(f"Reranker score: {chunk['reranker_score']:.4f}")
        print(f"Retrieved by: {chunk['retrieval_sources']}")
        print(f"Text: {chunk['text'][:500]}")


    print("\n=============================================")
