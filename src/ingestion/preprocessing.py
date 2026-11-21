
import re
from utils.logger import get_logger
logger = get_logger(__name__)

def preprocess_document(text):
    """
    Clean formatting inconsistencies while preserving legal content hence not preserving
    negatation words excluding from stop words to avoid accidently removing critical words.
    Original text should remain unchanged.
    """

    if not isinstance(text, str):
        raise TypeError("Document text must be a string")

    # Normalize Windows and Unix line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove trailing spaces from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Replace multiple blank lines with a maximum of two and tab
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove leading and trailing whitespace from the document
    cleaned_text = text.strip()

    logger.debug("Document preprocessing completed")

    return cleaned_text
