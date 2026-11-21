from sentence_transformers import CrossEncoder
from config.settings import QUERY_INSTRUCTION, RERANKER_MODEL
from retrieval.dense_retriever import DenseRetriever

# Original benchmark question and gold evidence chunk.
query = (
    "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; "
    "Does the document indicate that the Agreement does not grant the "
    "Receiving Party any rights to the Confidential Information?"
)
gold_chunk_id = (
    "contractnli\\CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt::chunk_0035"
)

RETRIEVAL_K = 250
RERANK_TOP_K = 20

# Initialise the dense retriever and retrieve the top 250 candidates.
dense_retriever = DenseRetriever()
dense_results = dense_retriever.retrieve(query, top_k=RETRIEVAL_K)

print(f"\nDense candidates retrieved: {len(dense_results)}")

# Check whether the gold chunk enters the candidate pool.
dense_gold_rank = next(
    (
        rank for rank, item in enumerate(dense_results, start=1)
        if item["chunk_id"] == gold_chunk_id
    ),
    None
)

print(
    f"Gold chunk dense rank: {dense_gold_rank}"
    if dense_gold_rank
    else "Gold chunk not found in dense top 250"
)

# Load the cross-encoder reranker.
reranker = CrossEncoder(RERANKER_MODEL)

# Score each query-chunk pair jointly using the cross-encoder.
query_chunk_pairs = [
    [query, item["text"]]
    for item in dense_results
]

reranker_scores = reranker.predict(
    query_chunk_pairs,
    batch_size=32,
    show_progress_bar=True
)

# Attach reranker scores and sort candidates by relevance.
for item, score in zip(dense_results, reranker_scores):
    item["reranker_score"] = float(score)

reranked_results = sorted(
    dense_results,
    key=lambda item: item["reranker_score"],
    reverse=True
)

final_results = reranked_results[:RERANK_TOP_K]

# Check whether the gold chunk survives reranking.
reranked_gold_rank = next(
    (
        rank for rank, item in enumerate(final_results, start=1)
        if item["chunk_id"] == gold_chunk_id
    ),
    None
)

print(f"\nReranked candidates selected: {len(final_results)}")

print(
    f"Gold chunk reranked position: {reranked_gold_rank}"
    if reranked_gold_rank
    else "Gold chunk not present in reranked top 20"
)

# Display the final 20 candidates with both retrieval and reranker scores.
print("\nFINAL TOP 20 AFTER RERANKING")

for rank, item in enumerate(final_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Dense score: {item['dense_score']:.6f}")
    print(f"Reranker score: {item['reranker_score']:.6f}")
    print(f"Text: {item['text'][:300]}")