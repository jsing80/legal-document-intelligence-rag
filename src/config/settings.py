
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_DIR = Path(os.getenv("RAG_SAMPLE_DIR", PROJECT_ROOT / "data"))

CORPUS_DIR = SAMPLE_DIR / "corpus"
BENCHMARK_DIR = SAMPLE_DIR / "benchmarks"

PROCESSED_DOCUMENTS_DIR = PROJECT_ROOT / "processed_documents"
PROCESSED_DOCUMENTS_PATH = PROCESSED_DOCUMENTS_DIR / "documents.jsonl"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

CHUNKS_DIR = PROJECT_ROOT / "chunks"
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"
CHUNK_MANIFEST_PATH = CHUNKS_DIR / "chunk_manifest.json"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 128
NORMALIZE_EMBEDDINGS = True

CHROMA_DB_DIR = PROJECT_ROOT / "vector_db"
CHROMA_COLLECTION_NAME = "legal_documents_bge_large_1500_200"
CHROMA_READ_BATCH_SIZE = 5000

MODEL_INFERENCE_BATCH_SIZE = 32

SPARSE_RETRIEVAL_LIBRARY = "bm25s"
BM25_INDEX_DIR = PROJECT_ROOT / "bm25_index"
BM25_CORPUS_PATH = BM25_INDEX_DIR / "corpus.jsonl"

DENSE_RETRIEVAL_K = 20
BM25_RETRIEVAL_K = 20
RRF_K = 60

# RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_MODEL= "BAAI/bge-reranker-v2-m3"
FINAL_CONTEXT_K = 10

QUERY_INSTRUCTION = ("Represent this sentence for searching relevant passages: ")

GENERATION_MODEL = "gemini-3.5-flash-lite"
GENERATION_MAX_OUTPUT_TOKENS = 2048

REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"

