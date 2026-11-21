
import json
import random

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from config.settings import (
    GENERATION_MODEL,
    GENERATION_MAX_OUTPUT_TOKENS,
    REPORT_DIR
)

from utils.logger import get_logger


logger = get_logger(__name__)

GENERATION_REPORT_PATH = (
    REPORT_DIR / "generation_evaluation_results.json"
)

RAGAS_REPORT_PATH = REPORT_DIR / "ragas_evaluation_results.json"

RANDOM_SEED = 42
MAX_RAGAS_SAMPLES = 20


def load_generation_results():
    """
    Load the results produced by generation_evaluator.py.

    Only successful, grounded generation results with a reference
    answer and retrieved contexts are eligible for RAGAS evaluation.
    """

    if not GENERATION_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Generation evaluation report not found. "
            "Run generation evaluation before RAGAS."
        )

    with GENERATION_REPORT_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        generation_results = json.load(file)

    valid_records = []

    for record in generation_results:

        # Exclude failed, rejected and no-context requests.
        if record.get("status") != "SUCCESS":
            continue

        answer = record.get("answer", "").strip()
        reference = record.get("ground_truth", "").strip()
        contexts = record.get("contexts", [])

        # RAGAS requires an answer, reference and retrieved context.
        if not answer or not reference or not contexts:
            continue

        valid_records.append({
            "user_input": record["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference
        })

    return valid_records


def evaluate_ragas():
    """
    Evaluate previously generated benchmark answers using RAGAS.

    This function does not invoke retrieval or answer generation.
    It only evaluates the saved generation results.
    """

    # -------------------------------------------------
    # STEP 1: Load existing generation results
    # -------------------------------------------------
    ragas_records = load_generation_results()

    if not ragas_records:
        raise RuntimeError(
            "No successful generation records available for RAGAS."
        )

    # Preserve the existing RAGAS sample size and reproducibility.
    rng = random.Random(RANDOM_SEED)

    if len(ragas_records) > MAX_RAGAS_SAMPLES:
        ragas_records = rng.sample(
            ragas_records,
            k=MAX_RAGAS_SAMPLES
        )

    logger.info(
        "Running RAGAS on %d saved generation results",
        len(ragas_records)
    )

    # -------------------------------------------------
    # STEP 2: Initialize the RAGAS evaluation judge
    # -------------------------------------------------
    # This Gemini model evaluates the saved answers.
    # It does not generate legal answers or retrieve documents.
    evaluator_model = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=0,
        max_output_tokens=GENERATION_MAX_OUTPUT_TOKENS
    )

    evaluator_llm = LangchainLLMWrapper(evaluator_model)

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
    )

    # -------------------------------------------------
    # STEP 3: Build the RAGAS dataset
    # -------------------------------------------------
    dataset = EvaluationDataset.from_list(ragas_records)

    # -------------------------------------------------
    # STEP 4: Run RAGAS metrics
    # -------------------------------------------------
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # -------------------------------------------------
    # STEP 5: Save evaluation results
    # -------------------------------------------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result_df = results.to_pandas()

    report = {
        "total_evaluated": len(ragas_records),
        "aggregate_scores": (
            result_df.mean(numeric_only=True).to_dict()
        ),
        "query_results": result_df.to_dict(orient="records")
    }

    with RAGAS_REPORT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # -------------------------------------------------
    # STEP 6: Print evaluation summary
    # -------------------------------------------------
    print("\n========== RAGAS EVALUATION ==========")
    print(f"Total evaluated queries: {len(ragas_records)}")

    print("\nAggregate Scores:")

    for metric, score in report["aggregate_scores"].items():
        print(f"{metric}: {score:.4f}")

    print(f"\nReport saved to: {RAGAS_REPORT_PATH}")
    print("======================================")

    logger.info("RAGAS evaluation completed.")

    return report


if __name__ == "__main__":
    evaluate_ragas()
