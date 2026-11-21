
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from config.settings import CORPUS_DIR
from evaluation.benchmark_loader import load_benchmarks
from evaluation.retrieval_evaluator import normalize_path
from retrieval.retrieval_pipeline import LegalRetrievalPipeline
from utils.logger import get_logger


logger = get_logger(__name__)

# Debug configuration
SAMPLE_SIZE = 110
RANDOM_SEED = 42

RETRIEVAL_STAGES = [
    "dense_results",
    "bm25_results",
    "fused_results",
    "final_results"
]

REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_PATH = REPORT_DIR / "retrieval_matching_debug.xlsx"


def get_benchmark_source_text(evidence):
    """
    Extract benchmark evidence using its original character offsets.

    The source document is resolved from the configured CORPUS_DIR.
    Invalid offsets are reported rather than silently accepted.
    """

    relative_path = Path(evidence["file_path"])
    source_path = (CORPUS_DIR / relative_path).resolve()

    start_char = evidence["start_char"]
    end_char = evidence["end_char"]

    expected_length = end_char - start_char

    if not source_path.exists():
        return {
            "resolved_path": str(source_path),
            "document_length": None,
            "expected_length": expected_length,
            "extracted_length": 0,
            "source_text": "",
            "status": "SOURCE FILE NOT FOUND"
        }

    source_text = source_path.read_text(encoding="utf-8")
    document_length = len(source_text)

    if start_char < 0 or end_char <= start_char:
        return {
            "resolved_path": str(source_path),
            "document_length": document_length,
            "expected_length": expected_length,
            "extracted_length": 0,
            "source_text": "",
            "status": "INVALID BENCHMARK OFFSETS"
        }

    if end_char > document_length:
        return {
            "resolved_path": str(source_path),
            "document_length": document_length,
            "expected_length": expected_length,
            "extracted_length": 0,
            "source_text": "",
            "status": "END OFFSET EXCEEDS DOCUMENT LENGTH"
        }

    extracted_text = source_text[start_char:end_char]

    extracted_length = len(extracted_text)

    if extracted_length != expected_length:
        status = "EXTRACTED LENGTH MISMATCH"
    else:
        status = "OK"

    return {
        "resolved_path": str(source_path),
        "document_length": document_length,
        "expected_length": expected_length,
        "extracted_length": extracted_length,
        "source_text": extracted_text,
        "status": status
    }


def get_chunk_content(retrieval_chunk):
    """Return retrieved chunk text."""
    return retrieval_chunk.get(
        "content",
        retrieval_chunk.get("text", "")
    )


def get_retrieved_metadata(retrieval_chunk):
    """Return metadata attached to a retrieved chunk."""
    return retrieval_chunk.get("metadata", {})


def collect_debug_results():
    """Run retrieval for sampled benchmark queries and collect debug evidence."""

    benchmark_records = load_benchmarks()

    rng = random.Random(RANDOM_SEED)

    sampled_records = rng.sample(
        benchmark_records,
        k=min(SAMPLE_SIZE, len(benchmark_records))
    )

    retrieval_pipeline = LegalRetrievalPipeline()

    benchmark_rows = []
    retrieved_rows = []
    path_comparison_rows = []

    for query_number, benchmark_record in enumerate(
        sampled_records,
        start=1
    ):

        benchmark_query = benchmark_record["query"]
        benchmark_file = benchmark_record["benchmark_file"]
        benchmark_evidence = benchmark_record["gold_evidence"]

        logger.info(
            "Debugging query %d of %d",
            query_number,
            len(sampled_records)
        )

        # Run retrieval for the benchmark query.
        retrieval_results = retrieval_pipeline.retrieve(benchmark_query)

        # --------------------------------------------------
        # Sheet 1: Benchmark evidence and source extraction
        # --------------------------------------------------

        for evidence_number, evidence in enumerate(
            benchmark_evidence,
            start=1
        ):

            extraction = get_benchmark_source_text(evidence)

            benchmark_rows.append([
                query_number,
                benchmark_file,
                benchmark_query,
                evidence_number,
                evidence["file_path"],
                extraction["resolved_path"],
                evidence["start_char"],
                evidence["end_char"],
                extraction["expected_length"],
                extraction["document_length"],
                extraction["extracted_length"],
                extraction["status"],
                extraction["source_text"]
            ])

        # --------------------------------------------------
        # Sheet 2: Retrieved chunks at every retrieval stage
        # --------------------------------------------------

        for retrieval_stage in RETRIEVAL_STAGES:

            retrieval_chunks = retrieval_results.get(
                retrieval_stage,
                []
            )

            for retrieval_rank, retrieval_chunk in enumerate(
                retrieval_chunks,
                start=1
            ):

                metadata = get_retrieved_metadata(retrieval_chunk)

                retrieved_chunk_id = retrieval_chunk.get(
                    "chunk_id",
                    ""
                )

                retrieved_source_path = metadata.get(
                    "source_path",
                    ""
                )

                retrieved_start = metadata.get(
                    "start_char",
                    ""
                )

                retrieved_end = metadata.get(
                    "end_char",
                    ""
                )

                retrieved_content = get_chunk_content(retrieval_chunk)

                retrieved_rows.append([
                    query_number,
                    benchmark_file,
                    benchmark_query,
                    retrieval_stage,
                    retrieval_rank,
                    retrieved_chunk_id,
                    retrieved_source_path,
                    retrieved_start,
                    retrieved_end,
                    len(retrieved_content),
                    retrieved_content
                ])

                # --------------------------------------------------
                # Sheet 3: Benchmark and retrieved source path comparison
                # --------------------------------------------------

                for evidence_number, evidence in enumerate(
                    benchmark_evidence,
                    start=1
                ):

                    benchmark_path = normalize_path(
                        evidence["file_path"]
                    )

                    retrieved_path = normalize_path(
                        retrieved_source_path
                    )

                    path_comparison_rows.append([
                        query_number,
                        benchmark_query,
                        evidence_number,
                        evidence["file_path"],
                        retrieval_stage,
                        retrieval_rank,
                        retrieved_chunk_id,
                        retrieved_source_path,
                        benchmark_path == retrieved_path
                    ])

    return (
        benchmark_rows,
        retrieved_rows,
        path_comparison_rows
    )


def format_worksheet(worksheet):
    """Apply basic formatting to the Excel worksheet."""

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for cell in worksheet[1]:
        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            start_color="1F4E78",
            end_color="1F4E78",
            fill_type="solid"
        )

    for column_cells in worksheet.columns:

        column_letter = get_column_letter(
            column_cells[0].column
        )

        max_length = max(
            len(str(cell.value or "")) if cell.data_type != "s"
            else min(len(str(cell.value or "")), 60)
            for cell in column_cells
        )

        worksheet.column_dimensions[column_letter].width = min(
            max(max_length + 2, 15),
            60
        )


def save_debug_report(
    benchmark_rows,
    retrieved_rows,
    path_comparison_rows
):
    """Save the three debug worksheets into an Excel workbook."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    workbook = Workbook()

    # --------------------------------------------------
    # Sheet 1: Benchmark Evidence
    # --------------------------------------------------

    benchmark_sheet = workbook.active
    benchmark_sheet.title = "Benchmark Evidence"

    benchmark_sheet.append([
        "Query Number",
        "Benchmark File",
        "Benchmark Query",
        "Evidence Number",
        "Benchmark Source File",
        "Resolved Source Path",
        "Start Character",
        "End Character",
        "Expected Evidence Length",
        "Source Document Length",
        "Extracted Text Length",
        "Extraction Status",
        "Benchmark Source Text"
    ])

    for row in benchmark_rows:
        benchmark_sheet.append(row)

    # --------------------------------------------------
    # Sheet 2: Retrieved Chunks
    # --------------------------------------------------

    retrieved_sheet = workbook.create_sheet(
        "Retrieved Chunks"
    )

    retrieved_sheet.append([
        "Query Number",
        "Benchmark File",
        "Benchmark Query",
        "Retrieval Stage",
        "Retrieval Rank",
        "Chunk ID",
        "Retrieved Source Path",
        "Retrieved Start Character",
        "Retrieved End Character",
        "Retrieved Content Length",
        "Retrieved Chunk Content"
    ])

    for row in retrieved_rows:
        retrieved_sheet.append(row)

    # --------------------------------------------------
    # Sheet 3: Path Comparison
    # --------------------------------------------------

    path_sheet = workbook.create_sheet(
        "Path Comparison"
    )

    path_sheet.append([
        "Query Number",
        "Benchmark Query",
        "Evidence Number",
        "Benchmark Source Path",
        "Retrieval Stage",
        "Retrieval Rank",
        "Chunk ID",
        "Retrieved Source Path",
        "Exact Path Match"
    ])

    for row in path_comparison_rows:
        path_sheet.append(row)

    # Format all worksheets.
    for worksheet in workbook.worksheets:
        format_worksheet(worksheet)

    workbook.save(REPORT_PATH)

    logger.info(
        "Debug report saved to %s",
        REPORT_PATH
    )

    print(f"\nDebug report saved to: {REPORT_PATH}")


def main():
    """Run retrieval debugging and generate the Excel report."""

    (
        benchmark_rows,
        retrieved_rows,
        path_comparison_rows
    ) = collect_debug_results()

    save_debug_report(
        benchmark_rows,
        retrieved_rows,
        path_comparison_rows
    )

    print(f"Benchmark evidence rows: {len(benchmark_rows)}")
    print(f"Retrieved chunk rows: {len(retrieved_rows)}")
    print(f"Path comparison rows: {len(path_comparison_rows)}")

    invalid_rows = [
        row for row in benchmark_rows
        if row[11] != "OK"
    ]

    print(f"Evidence extraction errors: {len(invalid_rows)}")

    if invalid_rows:
        print("\nEvidence extraction issues:")

        for row in invalid_rows:
            print(
                f"Query {row[0]}, Evidence {row[3]}: "
                f"{row[11]} | "
                f"Expected length: {row[8]} | "
                f"Extracted length: {row[10]}"
            )


if __name__ == "__main__":
    main()
