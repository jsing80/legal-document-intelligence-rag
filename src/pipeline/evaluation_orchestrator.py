
import json

from config.settings import REPORT_DIR

from evaluation.retrieval_evaluator import evaluate_retrieval
from evaluation.generation_evaluator import evaluate_generation
from evaluation.ragas_evaluator import evaluate_ragas

from utils.logger import get_logger


logger = get_logger(__name__)


class EvaluationOrchestrator:
    """
    Coordinates all offline evaluation activities.

    Execution order:
        1. Retrieval evaluation
        2. Generation evaluation
        3. RAGAS evaluation

    Each evaluator performs its own evaluation logic.
    This orchestrator coordinates execution and saves reports.
    """

    def __init__(self):
        # Create the reports directory if it does not exist.
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluation report directory: %s", REPORT_DIR)

    def run_retrieval_evaluation(self):
        """
        Run retrieval evaluation and save aggregate metrics as JSON.
        """

        logger.info("Starting retrieval evaluation.")

        # The existing evaluator calculates metrics for:
        # dense_results, bm25_results, fused_results and final_results.
        # It returns an aggregate dictionary.
        retrieval_report = evaluate_retrieval()

        # Save the returned dictionary as a JSON report.
        report_path = REPORT_DIR / "retrieval_evaluation_results.json"

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(
                retrieval_report,
                file,
                indent=4,
                ensure_ascii=False
            )

        logger.info(
            "Retrieval evaluation report saved to: %s",
            report_path
        )

        print(f"\nRetrieval report saved to: {report_path}")

        return retrieval_report

    def run_generation_evaluation(self):
        """
        Run benchmark questions through the online RAG orchestrator.

        The generation evaluator saves its results to:
        generation_evaluation_results.json
        """

        logger.info("Starting generation evaluation.")

        generation_results = evaluate_generation()

        successful_queries = sum(
            1
            for result in generation_results
            if result["status"] == "SUCCESS"
        )

        logger.info(
            "Generation evaluation completed: %d/%d successful.",
            successful_queries,
            len(generation_results)
        )

        return generation_results

    def run_ragas_evaluation(self):
        """
        Evaluate previously generated answers using RAGAS.

        RAGAS reads generation_evaluation_results.json.
        It does not repeat retrieval or answer generation.
        """

        logger.info("Starting RAGAS evaluation.")

        ragas_report = evaluate_ragas()

        logger.info("RAGAS evaluation completed.")

        return ragas_report

    def run_all(self):
        """
        Execute all offline evaluations sequentially.

        Execution stops if an evaluator raises an exception.
        """

        logger.info("Starting complete offline evaluation.")

        # STEP 1: Evaluate retrieval quality and save its report.
        retrieval_report = self.run_retrieval_evaluation()

        # STEP 2: Run guarded RAG for benchmark questions.
        # Save generated answers and contexts for RAGAS.
        generation_results = self.run_generation_evaluation()

        # STEP 3: Evaluate saved generation results using RAGAS.
        ragas_report = self.run_ragas_evaluation()

        # Prepare a consolidated execution summary.
        evaluation_summary = {
            "retrieval": retrieval_report,
            "generation": {
                "total_queries": len(generation_results),
                "successful_queries": sum(
                    1
                    for result in generation_results
                    if result["status"] == "SUCCESS"
                )
            },
            "ragas": ragas_report
        }

        logger.info("Complete offline evaluation finished.")

        return evaluation_summary


if __name__ == "__main__":

    # Initialize the offline evaluation orchestrator.
    evaluation_pipeline = EvaluationOrchestrator()

    # Execute retrieval, generation and RAGAS evaluations.
    evaluation_summary = evaluation_pipeline.run_all()

    # Print the final evaluation summary.
    print("\n========== OFFLINE EVALUATION SUMMARY ==========")

    print("\nRetrieval evaluation:")

    for stage, metrics in evaluation_summary["retrieval"].items():
        print(f"\n{stage.upper()}")

        for metric_name, score in metrics.items():
            print(f"{metric_name:20s}: {score:.4f}")

    print("\nGeneration evaluation:")
    print(
        "Total queries:",
        evaluation_summary["generation"]["total_queries"]
    )
    print(
        "Successful queries:",
        evaluation_summary["generation"]["successful_queries"]
    )

    print("\nRAGAS evaluation:")

    for metric_name, score in (
        evaluation_summary["ragas"]["aggregate_scores"].items()
    ):
        print(f"{metric_name:20s}: {score:.4f}")

    print("\nAll reports saved under:", REPORT_DIR)
    print("================================================")
