
import json

from dotenv import load_dotenv

from config.settings import PROJECT_ROOT, REPORT_DIR
from evaluation.benchmark_loader import load_benchmarks

# Load the project's API key before importing the online orchestrator.
# The orchestrator imports generator.py, which initializes the Gemini client.
load_dotenv(PROJECT_ROOT / ".env")

from pipeline.rag_orchestrator import RAGOrchestrator
from utils.logger import get_logger


logger = get_logger(__name__)

REPORT_PATH = REPORT_DIR / "generation_evaluation_results.json"
TEMP_REPORT_PATH = REPORT_DIR / "generation_evaluation_results.tmp"


def save_generation_results(generation_results):
    """
    Persist the current generation evaluation results.

    This function is called after every benchmark query.
    A temporary file is written first and then replaced atomically
    to reduce the risk of leaving an incomplete JSON report.
    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with TEMP_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            generation_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Replace the previous report only after the new JSON is complete.
    TEMP_REPORT_PATH.replace(REPORT_PATH)


def evaluate_generation():
    """
    Run all benchmark questions through the online RAG orchestrator.

    Execution for each query:
        Input Guardrail
            -> Retrieval
            -> Generation
            -> Output Guardrail

    Save the accumulated results after every query so that
    completed results are preserved if execution is interrupted.
    """

    # Load all benchmark questions. No sampling is performed.
    benchmark_records = load_benchmarks()

    total_queries = len(benchmark_records)

    logger.info(
        "Starting generation evaluation for %d benchmark queries.",
        total_queries
    )

    # Initialize the online RAG orchestrator once.
    # Its retrieval pipeline is reused across benchmark questions.
    rag_pipeline = RAGOrchestrator()

    generation_results = []

    # Process every benchmark question.
    for query_number, benchmark_record in enumerate(
        benchmark_records,
        start=1
    ):

        query = benchmark_record["query"]

        logger.info(
            "Generation evaluation: query %d/%d",
            query_number,
            total_queries
        )

        # Construct the benchmark reference answer from annotations.
        reference_answer = "\n".join(
            evidence["answer"].strip()
            for evidence in benchmark_record["gold_evidence"]
            if evidence.get("answer", "").strip()
        )

        try:
            # Run the complete guarded RAG pipeline once.
            rag_result = rag_pipeline.ask(query)

            # Preserve the answer, contexts, sources and execution status.
            result = {
                "benchmark_file": benchmark_record["benchmark_file"],
                "question": query,
                "answer": rag_result["answer"],
                "ground_truth": reference_answer,
                "contexts": rag_result["retrieved_contexts"],
                "retrieved_chunk_ids": [
                    source["chunk_id"]
                    for source in rag_result["sources"]
                    if source.get("chunk_id")
                ],
                "sources": rag_result["sources"],
                "status": rag_result["status"]
            }

            generation_results.append(result)

            logger.info(
                "Query %d/%d completed with status: %s",
                query_number,
                total_queries,
                rag_result["status"]
            )

        except Exception as error:

            # Record the failed query rather than losing the results
            # from previously completed benchmark questions.
            logger.exception(
                "Generation evaluation failed for query %d/%d",
                query_number,
                total_queries
            )

            generation_results.append({
                "benchmark_file": benchmark_record["benchmark_file"],
                "question": query,
                "answer": "",
                "ground_truth": reference_answer,
                "contexts": [],
                "retrieved_chunk_ids": [],
                "sources": [],
                "status": f"FAILED: {str(error)}"
            })

        # Save all completed results after every query.
        save_generation_results(generation_results)

        successful_queries = sum(
            1
            for record in generation_results
            if record["status"] == "SUCCESS"
        )

        logger.info(
            "Progress: %d/%d queries saved. Successful: %d",
            len(generation_results),
            total_queries,
            successful_queries
        )

        print(
            f"Generation evaluation: "
            f"{len(generation_results)}/{total_queries} completed. "
            f"Successful: {successful_queries}"
        )

    # Final summary after all benchmark questions are processed.
    successful_queries = sum(
        1
        for record in generation_results
        if record["status"] == "SUCCESS"
    )

    failed_or_rejected = total_queries - successful_queries

    print("\n========== GENERATION EVALUATION ==========")
    print(f"Total queries      : {total_queries}")
    print(f"Successful queries : {successful_queries}")
    print(f"Failed/rejected    : {failed_or_rejected}")
    print(f"Report saved to    : {REPORT_PATH}")
    print("===========================================")

    logger.info("Generation evaluation completed.")

    return generation_results


if __name__ == "__main__":
    evaluate_generation()
