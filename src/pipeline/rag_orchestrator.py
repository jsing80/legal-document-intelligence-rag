
from guardrails.input_guardrail import (
    validate_input,
    InputGuardrailError
)
from guardrails.output_guardrail import (
    validate_output,
    OutputGuardrailError
)

from retrieval.retrieval_pipeline import LegalRetrievalPipeline
from generation.generator import generate_answer
from utils.logger import get_logger


logger = get_logger(__name__)


class RAGOrchestrator:
    """
    Online orchestrator for a single legal-document RAG request.

    Execution sequence:
    Input Guardrail
        -> Retrieval
        -> Context Availability Check
        -> Answer Generation
        -> Output Guardrail
        -> Return Answer
    """

    def __init__(self):
        # Initialize the retrieval components once.
        # Dense retriever, BM25, fusion and reranker are managed
        # internally by LegalRetrievalPipeline.
        self.retrieval_pipeline = LegalRetrievalPipeline()

    def ask(self, user_question):
        """
        Process one user question through the complete RAG pipeline.

        Returns the answer, retrieved contexts, source information
        and execution status.
        """

        # -------------------------------------------------
        # STEP 1: Input guardrail
        # -------------------------------------------------
        try:
            # Validate and sanitize the incoming user question.
            # Use the validated question for all downstream steps.
            validated_question = validate_input(user_question)

        except InputGuardrailError as error:
            logger.warning(
                "Input rejected. Category: %s, Rule: %s",
                error.category,
                error.rule_id
            )

            return {
                "question": (
                    user_question
                    if isinstance(user_question, str)
                    else None
                ),
                "answer": (
                    "Your request could not be processed "
                    "because it did not pass input validation."
                ),
                "retrieved_contexts": [],
                "sources": [],
                "status": "INPUT_REJECTED",
                "guardrail": {
                    "stage": "input",
                    "category": error.category,
                    "rule_id": error.rule_id
                }
            }

        # -------------------------------------------------
        # STEP 2: Retrieval
        # -------------------------------------------------
        # The retrieval pipeline executes:
        # Dense Retrieval + BM25 -> Reciprocal Rank Fusion
        # -> Reranking.
        retrieval_results = self.retrieval_pipeline.retrieve(
            validated_question
        )

        # Only the final reranked chunks are passed to the generator.
        final_chunks = retrieval_results.get("final_results", [])

        # Remove chunks that do not contain usable text.
        final_chunks = [
            chunk for chunk in final_chunks
            if isinstance(chunk.get("text"), str)
            and chunk["text"].strip()
        ]

        # -------------------------------------------------
        # STEP 3: Context availability check
        # -------------------------------------------------
        # Do not call the LLM if retrieval did not return
        # usable evidence.
        if not final_chunks:
            logger.warning(
                "No usable retrieval context found for the question."
            )

            return {
                "question": validated_question,
                "answer": (
                    "I could not find relevant information in the "
                    "retrieved legal documents to answer this question."
                ),
                "retrieved_contexts": [],
                "sources": [],
                "status": "NO_CONTEXT",
                "guardrail": None
            }

        # Prepare the individual context strings for the response
        # and for subsequent offline evaluation.
        retrieved_contexts = []
        sources = []

        for rank, chunk in enumerate(final_chunks, start=1):
            metadata = chunk.get("metadata", {})

            retrieved_contexts.append(chunk["text"])

            sources.append({
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "source_path": metadata.get("source_path"),
                "start_char": metadata.get("start_char"),
                "end_char": metadata.get("end_char")
            })

        # -------------------------------------------------
        # STEP 4: Answer generation
        # -------------------------------------------------
        # Use the existing generator.py implementation.
        # It builds the evidence prompt and calls Gemini.
        answer = generate_answer(
            user_query=validated_question,
            retrieved_chunks=final_chunks
        )

        # -------------------------------------------------
        # STEP 5: Output guardrail
        # -------------------------------------------------
        # Validate the generated answer before returning it.
        try:
            validated_answer = validate_output(answer)

        except OutputGuardrailError as error:
            logger.warning(
                "Output rejected. Category: %s, Rule: %s",
                error.category,
                error.rule_id
            )

            return {
                "question": validated_question,
                "answer": (
                    "I could not generate a valid response "
                    "for this question. Please try again."
                ),
                "retrieved_contexts": retrieved_contexts,
                "sources": sources,
                "status": "OUTPUT_REJECTED",
                "guardrail": {
                    "stage": "output",
                    "category": error.category,
                    "rule_id": error.rule_id
                }
            }

        # -------------------------------------------------
        # STEP 6: Return the validated response
        # -------------------------------------------------
        logger.info("RAG request completed successfully.")

        return {
            "question": validated_question,
            "answer": validated_answer,
            "retrieved_contexts": retrieved_contexts,
            "sources": sources,
            "status": "SUCCESS",
            "guardrail": None
        }
