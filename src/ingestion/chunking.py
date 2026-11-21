
"""
Split processed documents into chunks and maintain them incrementally.
Only new or modified documents are re-chunked. Existing chunks for unchanged
documents are retained, while chunks for deleted documents are removed.
"""
import re
import hashlib
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

logger = get_logger(__name__)

# Prefer paragraph and line boundaries; use character-level splitting as fallback.
# SEPARATORS = ["\n\n", "\n", ""]
SEPARATORS = [""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    add_start_index=True
)

def create_chunk_metadata(document, chunk_index, start_char, chunk_text):
    """Build metadata associated with an individual legal document chunk."""
    return {
        "chunk_id": f"{document['document_id']}::chunk_{chunk_index:04d}",
        "document_id": document["document_id"],
        "source_file": document["source_file"],
        "source_path": document["source_path"],
        "chunk_index": chunk_index,
        "start_char": start_char,
        "end_char": start_char + len(chunk_text)
    }

def is_standalone_heading(text):
    """Identify short standalone legal headings without classifying ordinary sentences as headings."""
    text = text.strip()

    if not text or len(text) > 100 or "\n" in text:
        return False

    # Match numbered legal headings such as 5.7 No Warranty.
    numbered_heading = re.match(
        r"^\d+(?:\.\d+)*\.?\s+[A-Z].*$",
        text
    )

    # Match short uppercase headings such as LIMITATION OF WARRANTY.
    uppercase_heading = (
        len(text) <= 80
        and any(char.isalpha() for char in text)
        and text.rstrip(".").isupper()
        and not re.search(r"[,;:!?]", text)
    )

    # Avoid treating numbered sentences as headings.
    if numbered_heading and re.search(r"[.!?;]\s+\S", text):
        return False

    return bool(numbered_heading or uppercase_heading)


def chunk_documents(documents):
    """Split documents into chunks and attach source metadata."""
    all_chunks = []

    for document in documents:
        # Split the processed document text using the configured chunking strategy.
        split_documents = splitter.create_documents([document["processed_text"]])

        for index, split_document in enumerate(split_documents):
            # Extract the chunk text and its position in the original document.
            chunk_text = split_document.page_content
            start_char = split_document.metadata["start_index"]

            # Create metadata separately to keep chunk content and metadata independent.
            metadata = create_chunk_metadata(document, index, start_char, chunk_text)

            # Store the chunk text and its associated metadata together.
            all_chunks.append({
                "metadata": metadata,
                "text": chunk_text
            })

    # Log the number of chunks generated across all input documents.
    logger.info("Created %d chunks from %d documents", len(all_chunks), len(documents))
    return all_chunks

def get_content_hash(text):
    """Generate a SHA-256 hash to identify document content changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_chunking_config():
    """Return the settings used to generate chunks."""
    return {
        "strategy": "RecursiveCharacterTextSplitter",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        # "separators": SEPARATORS,
        #"heading_association": True,
        #"heading_detection_version": 1
    }

if __name__ == "__main__":
    from ingestion.document_persistence import load_persisted_documents
    from ingestion.chunk_persistence import load_chunks, save_chunks
    from ingestion.chunk_manifest import load_manifest, save_manifest
    # Load the current processed documents, previously saved chunks and manifest.
    documents = load_persisted_documents()
    existing_chunks = load_chunks()
    manifest = load_manifest()
    current_config = get_chunking_config()
    previous_documents = manifest["documents"]

    # A change in chunking settings invalidates all previously generated chunks.
    config_changed = manifest["chunking_config"] != current_config
    current_document_ids = {doc["document_id"] for doc in documents}
    documents_to_chunk = []
    updated_manifest = {}

    # Compare current document content against the previous processing record.
    for document in documents:
        document_id = document["document_id"]
        content_hash = get_content_hash(document["processed_text"])
        previous = previous_documents.get(document_id)

        # Skip documents whose content and chunking configuration are unchanged.
        if not config_changed and previous and previous["content_hash"] == content_hash:
            updated_manifest[document_id] = previous
        else:
            # New or modified documents must be chunked again.
            documents_to_chunk.append(document)
            updated_manifest[document_id] = {
                "content_hash": content_hash,
                "chunk_count": 0
            }

    changed_document_ids = {doc["document_id"] for doc in documents_to_chunk}

    logger.info(
        "Documents: %d | Unchanged: %d | New/modified: %d",
        len(documents),
        len(documents) - len(documents_to_chunk),
        len(documents_to_chunk)
    )

    # Generate chunks only for new or modified documents.
    new_chunks = chunk_documents(documents_to_chunk)

    # Keep unchanged chunks and discard chunks for modified or deleted documents.
    merged_chunks = [
        chunk for chunk in existing_chunks
        if chunk["metadata"]["document_id"] not in changed_document_ids
        and chunk["metadata"]["document_id"] in current_document_ids
    ]

    # Add the newly generated chunks to the retained chunks.
    merged_chunks.extend(new_chunks)

    # Record the latest chunk count for each new or modified document.
    new_chunk_counts = Counter(chunk["metadata"]["document_id"] for chunk in new_chunks)

    for document_id in changed_document_ids:
        updated_manifest[document_id]["chunk_count"] = new_chunk_counts[document_id]

    # Save the updated chunks first, followed by their processing manifest.
    save_chunks(merged_chunks)

    save_manifest({
        "chunking_config": current_config,
        "documents": updated_manifest
    })

    logger.info(
        "Chunking complete | New chunks: %d | Total chunks: %d",
        len(new_chunks),
        len(merged_chunks)
    )
