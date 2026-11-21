
from config.settings import RRF_K


def reciprocal_rank_fusion(dense_results, bm25_results, top_k=20):
    """Combine dense and BM25 rankings using Reciprocal Rank Fusion."""

    # Maintain one candidate per chunk ID, even if both retrievers find it.
    fused_candidates = {}

    # Process the dense and BM25 ranked lists independently.
    for source, results in [
        ("dense", dense_results),
        ("bm25", bm25_results)
    ]:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]

            # Calculate the RRF contribution using the candidate's rank.
            rrf_score = 1 / (RRF_K + rank)

            if chunk_id not in fused_candidates:
                fused_candidates[chunk_id] = {
                    **chunk,
                    "rrf_score": 0.0,
                    "retrieval_sources": []
                }

            # Accumulate the contribution from each retrieval method.
            fused_candidates[chunk_id]["rrf_score"] += rrf_score
            fused_candidates[chunk_id]["retrieval_sources"].append(source)

    # Rank candidates by their combined RRF score.
    ranked_candidates = sorted(
        fused_candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_candidates[:top_k]
