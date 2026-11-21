import json
import bm25s

from config.settings import BM25_INDEX_DIR, BM25_CORPUS_PATH
from ingestion.chunk_persistence import load_chunks
from utils.logger import get_logger

logger = get_logger(__name__)


def build_bm25_index():
    """Build and persist a BM25 index from the existing legal chunks."""

    # Load the same chunks used to generate dense embeddings.
    chunks = load_chunks()
    if not chunks:
        raise ValueError("No source chunks found for BM25 indexing.")

    # Preserve original chunk IDs and metadata for result mapping.
    corpus = [
        {
            "chunk_id": chunk["metadata"]["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        }
        for chunk in chunks
    ]

    bm25_corpus = [item["text"] for item in corpus]

    # Return token strings so BM25S builds a string-keyed vocabulary.
    corpus_tokens = bm25s.tokenize(bm25_corpus, return_ids=False)

    # Build one BM25 search document per original chunk.
    bm25_index = bm25s.BM25(method="lucene")
    bm25_index.index(corpus_tokens)

    # Persist the BM25 index and its vocabulary.
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25_index.save(str(BM25_INDEX_DIR))

    # Persist original chunks in the same order as the BM25 index.
    with BM25_CORPUS_PATH.open("w", encoding="utf-8") as file:
        for item in corpus:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("BM25 index created for %d chunks", len(corpus))


if __name__ == "__main__":
    build_bm25_index()