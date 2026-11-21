
import json
import bm25s
# from bm25s.tokenization import Tokenizer
from config.settings import BM25_INDEX_DIR, BM25_CORPUS_PATH


class BM25Retriever:
    """Retrieve chunks using keyword-based BM25 ranking."""

    def __init__(self):
        # Load the persisted BM25 index using memory mapping.
        self.bm25_index = bm25s.BM25.load(str(BM25_INDEX_DIR),mmap=True)

        # Load the ordered corpus to map BM25 positions to chunk records.
        with BM25_CORPUS_PATH.open("r", encoding="utf-8") as file:
            self.corpus = [json.loads(line) for line in file if line.strip()]

    def retrieve(self, query, top_k=10):
        """Tokenize the query and retrieve the highest-scoring chunks."""

        # Tokenize the query using the persisted corpus vocabulary.
        query_tokens = bm25s.tokenize([query], return_ids=False)

        # Retrieve the highest-scoring corpus positions and their BM25 scores.
        document_indices, scores = self.bm25_index.retrieve(query_tokens,k=top_k)

        retrieved_chunks = []

        # Map each returned position to its original chunk and metadata.
        for index, score in zip(document_indices[0], scores[0]):
            if index < 0:
                continue

            chunk = self.corpus[int(index)]

            retrieved_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "bm25_score": float(score)
            })

        return retrieved_chunks
