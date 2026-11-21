
import json

from config.settings import PROCESSED_DOCUMENTS_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

def save_documents(documents, output_path=PROCESSED_DOCUMENTS_PATH):
    """Persist document records as JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for document in documents:
            json.dump(document, file, ensure_ascii=False)
            file.write("\n")

    logger.info(
        "Persisted %d documents to %s",
        len(documents),
        output_path
    )


def load_persisted_documents(input_path=PROCESSED_DOCUMENTS_PATH):
    """Load document records from a JSONL file."""

    if not input_path.exists():
        raise FileNotFoundError(f"Persisted documents not found: {input_path}")

    documents = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    logger.info("Loaded %d persisted documents", len(documents))

    return documents
