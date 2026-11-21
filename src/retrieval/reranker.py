
from sentence_transformers import CrossEncoder
from config.settings import RERANKER_MODEL


class LegalReranker:
    """Rerank retrieved legal chunks using query-document relevance."""

    def __init__(self):
        # Load the cross-encoder once and reuse it across queries.
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query, candidates, top_n=5):
        """Score query-chunk pairs and return the highest-ranked chunks."""

        if not candidates:
            return []

        # A cross-encoder jointly evaluates the query and each candidate chunk.
        query_chunk_pairs = [[query, chunk["text"]] for chunk in candidates]
        
        # Higher model scores indicate greater query-document relevance.
        scores = self.model.predict(query_chunk_pairs)

        # Attach each relevance score to its corresponding candidate.
        for chunk, score in zip(candidates, scores):
            chunk["reranker_score"] = float(score)

        # Sort by cross-encoder relevance, not by the original RRF score.
        reranked_chunks = sorted(
            candidates,
            key=lambda item: item["reranker_score"],
            reverse=True
        )

        return reranked_chunks[:top_n]
