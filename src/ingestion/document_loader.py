
from pathlib import Path

from config.settings import CORPUS_DIR
from utils.logger import get_logger
from ingestion.preprocessing import preprocess_document

logger = get_logger(__name__)

def load_documents(corpus_dir=CORPUS_DIR):
    """
    Load legal text files and retain both original and processed content.
    """

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    documents = []

    # Read all text files, including nested folders
    txt_files = sorted(corpus_dir.rglob("*.txt"))

    logger.info("Found %d text files in corpus", len(txt_files))

    for file_path in txt_files:
        try:
            # Read original content without modifying the source file
            original_text = file_path.read_text(encoding="utf-8")

            if not original_text.strip():
                logger.warning("Skipping empty file: %s", file_path)
                continue

            # Create a cleaned copy for downstream processing
            processed_text = preprocess_document(original_text)

            documents.append({
                "document_id": str(file_path.relative_to(corpus_dir)),
                "source_file": file_path.name,
                "source_path": str(file_path),
                "text": original_text,
                "processed_text": processed_text
            })

        except UnicodeDecodeError:
            logger.warning("Could not decode file: %s", file_path)

    logger.info("Successfully loaded %d documents", len(documents))

    return documents
