
from collections import defaultdict
import random

from evaluation.benchmark_loader import load_benchmarks
from retrieval.retrieval_pipeline import LegalRetrievalPipeline
from utils.logger import get_logger


MAX_QUERIES_PER_FILE = 50
RANDOM_SEED = 42

logger = get_logger(__name__)

RETRIEVAL_STAGES = [
    "dense_results",
    "bm25_results",
    "fused_results",
    "final_results"
]


def normalize_path(file_path):
    """Make benchmark and retrieved document paths comparable."""
    normalized_path = str(file_path).replace("\\", "/").lower()
    corpus_marker = "/corpus/"

    if corpus_marker in normalized_path:
        normalized_path = normalized_path.split(corpus_marker, 1)[1]

    return normalized_path.lstrip("/")


def sample_benchmark_records(benchmark_records):
    """Select a reproducible sample of queries from each benchmark file."""
    records_by_file = defaultdict(list)

    for record in benchmark_records:
        records_by_file[record["benchmark_file"]].append(record)

    rng = random.Random(RANDOM_SEED)
    sampled_records = []

    for benchmark_file, records in sorted(records_by_file.items()):
        selected_records = rng.sample(
            records,
            k=min(MAX_QUERIES_PER_FILE, len(records))
        )
        sampled_records.extend(selected_records)

        logger.info(
            "Selected %d of %d queries from %s",
            len(selected_records),
            len(records),
            benchmark_file
        )

    return sampled_records


def prepare_benchmark_evidence(benchmark_evidence):
    """Organize expected evidence by document path."""
    benchmark_evidence_by_file = defaultdict(list)

    for evidence_index, evidence in enumerate(benchmark_evidence):
        benchmark_source_path = normalize_path(evidence["file_path"])

        benchmark_evidence_by_file[benchmark_source_path].append({
            "evidence_id": evidence_index,
            "start_char": evidence["start_char"],
            "end_char": evidence["end_char"]
        })

    return benchmark_evidence_by_file


def retrieve_benchmark_query(retrieval_pipeline, benchmark_query):
    """Run one benchmark question through the existing retrieval pipeline."""
    return retrieval_pipeline.retrieve(benchmark_query)


def match_retrieved_chunks(retrieval_chunks, benchmark_evidence_by_file):
    """Match retrieved chunks against expected evidence."""
    matched_evidence_ids = set()
    relevant_chunk_ids = set()
    retrieved_spans_by_evidence = defaultdict(list)
    first_relevant_rank = 0

    for retrieval_rank, retrieval_chunk in enumerate(
        retrieval_chunks,
        start=1
    ):
        retrieval_metadata = retrieval_chunk.get("metadata", {})
        retrieval_source_path = normalize_path(
            retrieval_metadata.get("source_path", "")
        )

        if retrieval_source_path not in benchmark_evidence_by_file:
            continue

        retrieval_start = retrieval_metadata.get("start_char")
        retrieval_end = retrieval_metadata.get("end_char")

        if retrieval_start is None or retrieval_end is None:
            continue

        for benchmark_passage in benchmark_evidence_by_file[
            retrieval_source_path
        ]:
            overlap_start = max(
                retrieval_start,
                benchmark_passage["start_char"]
            )
            overlap_end = min(
                retrieval_end,
                benchmark_passage["end_char"]
            )

            if overlap_start < overlap_end:
                evidence_id = benchmark_passage["evidence_id"]

                matched_evidence_ids.add(evidence_id)
                relevant_chunk_ids.add(retrieval_chunk["chunk_id"])

                retrieved_spans_by_evidence[evidence_id].append(
                    (overlap_start, overlap_end)
                )

                if first_relevant_rank == 0:
                    first_relevant_rank = retrieval_rank

    return {
        "matched_evidence_ids": matched_evidence_ids,
        "relevant_chunk_ids": relevant_chunk_ids,
        "retrieved_spans_by_evidence": retrieved_spans_by_evidence,
        "first_relevant_rank": first_relevant_rank
    }


def calculate_evidence_coverage(
    benchmark_evidence,
    retrieved_spans_by_evidence
):
    """Calculate percentage of expected evidence characters retrieved."""
    total_benchmark_chars = 0
    total_retrieved_chars = 0

    for evidence_index, evidence in enumerate(benchmark_evidence):
        benchmark_start = evidence["start_char"]
        benchmark_end = evidence["end_char"]

        total_benchmark_chars += benchmark_end - benchmark_start

        retrieved_spans = retrieved_spans_by_evidence.get(
            evidence_index,
            []
        )

        if not retrieved_spans:
            continue

        retrieved_spans.sort()
        merged_start, merged_end = retrieved_spans[0]

        for span_start, span_end in retrieved_spans[1:]:
            if span_start <= merged_end:
                merged_end = max(merged_end, span_end)
            else:
                total_retrieved_chars += merged_end - merged_start
                merged_start, merged_end = span_start, span_end

        total_retrieved_chars += merged_end - merged_start

    if total_benchmark_chars == 0:
        return 0.0

    return total_retrieved_chars / total_benchmark_chars


def calculate_retrieval_metrics(retrieval_chunks, benchmark_evidence):
    """Calculate retrieval metrics for one question and one retrieval stage."""

    benchmark_evidence_by_file = prepare_benchmark_evidence(
        benchmark_evidence
    )

    match_result = match_retrieved_chunks(
        retrieval_chunks,
        benchmark_evidence_by_file
    )

    total_benchmark_passages = len(benchmark_evidence)
    matched_passages = len(match_result["matched_evidence_ids"])

    hit_rate = 1.0 if matched_passages else 0.0

    passage_recall = (
        matched_passages / total_benchmark_passages
        if total_benchmark_passages else 0.0
    )

    precision = (
        len(match_result["relevant_chunk_ids"]) / len(retrieval_chunks)
        if retrieval_chunks else 0.0
    )

    evidence_recall = calculate_evidence_coverage(
        benchmark_evidence,
        match_result["retrieved_spans_by_evidence"]
    )

    mrr = (
        1.0 / match_result["first_relevant_rank"]
        if match_result["first_relevant_rank"] else 0.0
    )

    return {
        "hit_rate": hit_rate,
        "passage_recall": passage_recall,
        "precision": precision,
        "evidence_recall": evidence_recall,
        "mrr": mrr
    }


def evaluate_retrieval():
    """Evaluate benchmark questions across all retrieval stages."""

    benchmark_records = sample_benchmark_records(load_benchmarks())

    logger.info(
        "Running retrieval evaluation on %d benchmark queries",
        len(benchmark_records)
    )

    retrieval_pipeline = LegalRetrievalPipeline()

    query_level_results = {
        stage: [] for stage in RETRIEVAL_STAGES
    }

    for benchmark_index, benchmark_record in enumerate(
        benchmark_records,
        start=1
    ):
        benchmark_query = benchmark_record["query"]
        benchmark_evidence = benchmark_record["gold_evidence"]

        logger.info(
            "Evaluating benchmark query %d of %d",
            benchmark_index,
            len(benchmark_records)
        )

        retrieval_results = retrieve_benchmark_query(
            retrieval_pipeline,
            benchmark_query
        )

        for retrieval_stage in RETRIEVAL_STAGES:
            retrieval_chunks = retrieval_results.get(
                retrieval_stage,
                []
            )

            stage_metrics = calculate_retrieval_metrics(
                retrieval_chunks,
                benchmark_evidence
            )

            query_level_results[retrieval_stage].append(stage_metrics)

    evaluation_report = {}

    for retrieval_stage, query_results in query_level_results.items():
        if not query_results:
            continue

        evaluation_report[retrieval_stage] = {
            metric_name: sum(
                result[metric_name] for result in query_results
            ) / len(query_results)
            for metric_name in query_results[0]
        }

    return evaluation_report


if __name__ == "__main__":
    report = evaluate_retrieval()

    print("\nRETRIEVAL EVALUATION RESULTS")
    print("=" * 50)

    for retrieval_stage, metrics in report.items():
        print(f"\n{retrieval_stage.upper()}")

        for metric_name, score in metrics.items():
            print(
                f"{metric_name.replace('_', ' ').title():20s}: "
                f"{score:.4f}"
            )
