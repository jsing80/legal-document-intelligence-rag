import json

from config.settings import CHUNK_MANIFEST_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

def load_manifest(input_path=CHUNK_MANIFEST_PATH):
    """Load the record of documents processed by the chunking pipeline."""

    # Start with an empty manifest when the pipeline runs for the first time.
    if not input_path.exists():
        return {"chunking_config": None, "documents": {}}

    with input_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    logger.info("Loaded manifest for %d documents", len(manifest["documents"]))
    return manifest

def save_manifest(manifest, output_path=CHUNK_MANIFEST_PATH):
    """Persist document processing status and chunking configuration."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    # Write to a temporary file first to avoid leaving a partially written manifest.
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    # Replace the previous manifest only after the write completes.
    temp_path.replace(output_path)

    logger.info("Saved manifest for %d documents", len(manifest["documents"]))