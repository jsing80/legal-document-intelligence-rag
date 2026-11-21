
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from config.settings import (
    GENERATION_MODEL,
    GENERATION_MAX_OUTPUT_TOKENS
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Load the API key from the project's .env file.
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from the environment.")

# Initialize the Gemini client once when this module is loaded.
client = genai.Client(api_key=GEMINI_API_KEY)


def build_context(retrieved_chunks):
    """Format reranked chunks with source information for the LLM."""

    context_parts = []

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})

        # Preserve source and chunk identifiers so the answer can cite evidence.
        source = metadata.get("source_path", "Unknown source")
        chunk_id = chunk.get("chunk_id", "Unknown chunk")

        context_parts.append(
            f"[Evidence {rank}]\n"
            f"Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def build_prompt(user_query, context):
    """Create instructions that restrict the answer to retrieved legal evidence."""

    return f"""
You are a legal document analysis assistant.

Answer the user's question using only the evidence provided below.

Instructions:
1. Base every factual statement on the supplied evidence.
2. Do not use external knowledge or invent missing contractual terms.
3. If the evidence does not answer the question, explicitly state that
   the retrieved passages are insufficient.
4. Distinguish contractual obligations, permissions, prohibitions and exceptions.
5. Explain relevant exceptions or conditions when they appear in the evidence.
6. Cite supporting evidence using its source path and chunk ID.
7. Do not present your response as legal advice.

Retrieved evidence:
{context}

User question:
{user_query}

Provide a clear, structured answer with supporting citations.
""".strip()

def generate_answer(user_query, retrieved_chunks):
    """Generate a grounded answer from the top reranked legal document chunks."""

    if not retrieved_chunks:
        logger.warning("No retrieved evidence available for generation.")
        return "I could not retrieve relevant evidence to answer this question."

    # Convert the five reranked chunks into the context supplied to Gemini.
    context = build_context(retrieved_chunks)

    # Combine the evidence with the question and grounding instructions.
    prompt = build_prompt(user_query, context)

    logger.info(
        "Generating answer using %d retrieved chunks and model %s",
        len(retrieved_chunks),
        GENERATION_MODEL
    )

    # Call Gemini with controlled generation settings.
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=GENERATION_MAX_OUTPUT_TOKENS
        )
    )

    if not response.text:
        logger.warning("Gemini returned an empty response.")
        return "The model did not return an answer. Please retry."

    logger.info("Answer generation completed.")
    return response.text
