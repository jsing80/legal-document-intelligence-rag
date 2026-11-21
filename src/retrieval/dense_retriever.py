
from sentence_transformers import SentenceTransformer
from config.settings import (
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    QUERY_INSTRUCTION
)
import chromadb


class DenseRetriever:
    """Retrieve semantically similar chunks from ChromaDB."""

    def __init__(self):
        # Load the same embedding model used when creating stored vectors.
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to the existing persistent vector database.
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

    def retrieve(self, query, top_k=10):
        """Embed the query and retrieve the nearest chunks from ChromaDB."""

        # Apply BGE's retrieval instruction to the search query.
        query_text = QUERY_INSTRUCTION + query

        # Generate a normalized query vector in the same embedding space.
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Search ChromaDB using cosine distance.
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []

        # Convert ChromaDB's response into a consistent result structure.
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata or {},
                "dense_score": 1.0 - distance
            })

        return retrieved_chunks
