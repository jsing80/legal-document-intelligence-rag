
import json

from config.settings import BENCHMARK_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def load_benchmarks():
    benchmark_files = sorted(BENCHMARK_DIR.glob("*.json"))

    if not benchmark_files:
        raise FileNotFoundError(
            f"No benchmark JSON files found in {BENCHMARK_DIR}"
        )

    benchmark_records = []

    for file_path in benchmark_files:
        with file_path.open("r", encoding="utf-8") as file:
            try:
                benchmark_data = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in benchmark file: {file_path.name}. "
                    f"Error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
                ) from exc

        for test_index, test in enumerate(
            benchmark_data.get("tests", []), start=1
        ):
            query = test.get("query")
            snippets = test.get("snippets", [])

            # A benchmark question without both a query and gold evidence
            # cannot be used to measure retrieval quality.
            if not query or not snippets:
                logger.warning(
                    "Skipping incomplete test %d in %s",
                    test_index,
                    file_path.name,
                )
                continue

            gold_evidence = []

            for snippet in snippets:
                file_path_value = snippet.get("file_path")
                span = snippet.get("span")

                if not file_path_value or not isinstance(span, list) or len(span) != 2:
                    raise ValueError(
                        f"Invalid evidence in {file_path.name}, "
                        f"test {test_index}: expected file_path and span [start, end]."
                    )

                start_char, end_char = span

                # Reject invalid offsets here so they don't silently distort
                # passage recall or character-level evidence recall later.
                if (
                    not isinstance(start_char, int)
                    or isinstance(start_char, bool)
                    or not isinstance(end_char, int)
                    or isinstance(end_char, bool)
                    or start_char < 0
                    or end_char <= start_char
                ):
                    raise ValueError(
                        f"Invalid character span {span} in "
                        f"{file_path.name}, test {test_index}."
                    )

                gold_evidence.append({
                    "file_path": file_path_value,
                    "start_char": start_char,
                    "end_char": end_char,
                    "answer": snippet.get("answer", ""),
                })

            benchmark_records.append({
                "benchmark_file": file_path.name,
                "query": query,
                "gold_evidence": gold_evidence,
            })

    if not benchmark_records:
        raise ValueError("No valid benchmark questions were loaded.")

    logger.info(
        "Loaded %d benchmark queries from %d files.",
        len(benchmark_records),
        len(benchmark_files),
    )

    return benchmark_records


if __name__ == "__main__":
    benchmarks = load_benchmarks()

    print(f"Total benchmark queries: {len(benchmarks)}")

    if benchmarks:
        print("\nFirst benchmark record:")
        print(json.dumps(benchmarks[0], indent=2, ensure_ascii=False))
