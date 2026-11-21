import numpy as np
from config.settings import QUERY_INSTRUCTION
from retrieval.dense_retriever import DenseRetriever

# Original benchmark question and target document.
query = (
    "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; "
    "Does the document indicate that the Agreement does not grant the "
    "Receiving Party any rights to the Confidential Information?"
)
target_source_file = "CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt"
gold_chunk_indices = [35, 36]

# Load the existing embedding model and ChromaDB collection.
retriever = DenseRetriever()

# Inspect sample records to verify the actual stored ID format.
sample_records = retriever.collection.get(
    limit=5,
    include=["metadatas"]
)

print("\nSAMPLE CHROMA RECORDS")
for chunk_id, metadata in zip(sample_records["ids"], sample_records["metadatas"]):
    print(f"ID: {chunk_id}")
    print(f"Metadata: {metadata}")

# Find all chunks belonging to the target document using metadata.
target_records = retriever.collection.get(
    where={"source_file": target_source_file},
    include=["metadatas", "documents", "embeddings"]
)

print(f"\nChunks found for target document: {len(target_records['ids'])}")

# Identify the gold chunks using their stored chunk_index metadata.
gold_chunks = {}

for chunk_id, metadata, text, embedding in zip(
    target_records["ids"],
    target_records["metadatas"],
    target_records["documents"],
    target_records["embeddings"]
):
    if metadata.get("chunk_index") in gold_chunk_indices:
        gold_chunks[metadata["chunk_index"]] = {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": np.asarray(embedding)
        }

print("\nGOLD CHUNK DETAILS")
for chunk_index in gold_chunk_indices:
    if chunk_index in gold_chunks:
        item = gold_chunks[chunk_index]
        print(f"\nChunk index: {chunk_index}")
        print(f"Actual Chroma ID: {item['chunk_id']}")
        print(f"Text: {item['text']}")
    else:
        print(f"\nChunk index {chunk_index} was not found.")

if not gold_chunks:
    raise ValueError(
        "Neither gold chunk was found. Check the source_file and chunk_index metadata."
    )

# Generate the query embedding using the same instruction and normalization as retrieval.
query_embedding = retriever.model.encode(
    [QUERY_INSTRUCTION + query],
    normalize_embeddings=True,
    convert_to_numpy=True
)[0]

# Calculate cosine similarity between the query and each gold chunk.
print("\nGOLD CHUNK SIMILARITY")
for chunk_index, item in gold_chunks.items():
    score = float(np.dot(query_embedding, item["embedding"]))
    item["similarity"] = score
    print(f"Chunk {chunk_index}: {score:.6f}")

# Retrieve the top 100 chunks using the same query embedding.
results = retriever.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=100,
    include=["documents", "metadatas", "distances"]
)

# Convert cosine distances into similarity scores.
top_results = []

for chunk_id, text, metadata, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    top_results.append({
        "chunk_id": chunk_id,
        "score": 1.0 - distance,
        "metadata": metadata or {},
        "text": text
    })

# Find the actual rank of each gold chunk.
print("\nGOLD CHUNK RANK")
for chunk_index, item in gold_chunks.items():
    gold_rank = next(
        (
            rank for rank, result in enumerate(top_results, start=1)
            if result["chunk_id"] == item["chunk_id"]
        ),
        None
    )
    print(
        f"Chunk {chunk_index}: Rank {gold_rank}"
        if gold_rank
        else f"Chunk {chunk_index}: Not in top 100"
    )

# Display the top 10 competing chunks and their similarity scores.
print("\nTOP 10 DENSE RESULTS")

for rank, item in enumerate(top_results[:10], start=1):
    print(f"\nRank: {rank} | Score: {item['score']:.6f}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Source: {item['metadata'].get('source_file')}")
    print(f"Chunk index: {item['metadata'].get('chunk_index')}")
    print(f"Text: {item['text'][:300]}")

# Display the similarity range across the retrieved top 100.
print("\nTOP 100 SCORE RANGE")
print(f"Highest: {top_results[0]['score']:.6f}")
print(f"Lowest:  {top_results[-1]['score']:.6f}")