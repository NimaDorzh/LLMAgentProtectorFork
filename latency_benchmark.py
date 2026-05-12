import csv
import random
import statistics
import time
from pathlib import Path

from llmagentprotector import PolymorphicPromptAssembler


ITERATIONS = 1000
SEED = 1337
SYSTEM_PROMPT = "Please summarize the user-provided text."
TASK_TOPIC = "summarize the user-provided text"
USER_INPUT = (
    "Half Moon Bay is a picturesque coastal town in Northern California, located about 30 miles south of San "
    "Francisco. Known for its ocean views, beaches, and historic downtown, it is a popular destination for visitors."
)
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "latency_benchmark.csv"
BENCHMARKS = (
    "separator_generation_time",
    "prompt_assembly_time",
)
MODES = (
    "static",
    "dynamic",
)


def _build_protector(mode: str, seed: int | None = None) -> PolymorphicPromptAssembler:
    rng = random.Random(seed) if mode == "static" else None
    return PolymorphicPromptAssembler(
        SYSTEM_PROMPT,
        TASK_TOPIC,
        separator_mode=mode,
        rng=rng,
    )


def _time_once(func) -> float:
    start = time.perf_counter()
    func()
    return (time.perf_counter() - start) * 1000.0


def _benchmark_separator_generation(mode: str, iterations: int, seed: int) -> list[float]:
    protector = _build_protector(mode, seed=seed)
    timings_ms = []

    for iteration in range(iterations):
        session_id = f"latency-benchmark-{mode}-{iteration}"
        timings_ms.append(
            _time_once(lambda: protector._select_separators(session_id=session_id))
        )

    return timings_ms


def _benchmark_prompt_assembly(mode: str, iterations: int, seed: int) -> list[float]:
    protector = _build_protector(mode, seed=seed)
    timings_ms = []

    for iteration in range(iterations):
        session_id = f"latency-benchmark-{mode}-{iteration}"
        timings_ms.append(
            _time_once(
                lambda: protector.double_prompt_assemble(
                    user_input=USER_INPUT,
                    session_id=session_id,
                )
            )
        )

    return timings_ms


def _summarize(timings_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(timings_ms),
        "std_ms": statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
    }


def _write_rows(rows: list[dict[str, str | int | float]], results_file: Path) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark", "mode", "iteration", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary_tables(summaries: dict[str, dict[str, dict[str, float]]]) -> None:
    for benchmark in BENCHMARKS:
        print(benchmark)
        print("mode | mean_ms | std_ms | min_ms | max_ms")
        for mode in MODES:
            stats = summaries[benchmark][mode]
            print(
                f"{mode} | {stats['mean_ms']:.6f} | {stats['std_ms']:.6f} | {stats['min_ms']:.6f} | {stats['max_ms']:.6f}"
            )
        print()


def run_benchmark(
    iterations: int = ITERATIONS,
    seed: int = SEED,
    results_file: Path = RESULTS_FILE,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, str | int | float]]]:
    summaries: dict[str, dict[str, dict[str, float]]] = {
        benchmark: {} for benchmark in BENCHMARKS
    }
    rows: list[dict[str, str | int | float]] = []

    for mode in MODES:
        separator_timings = _benchmark_separator_generation(mode, iterations, seed)
        prompt_timings = _benchmark_prompt_assembly(mode, iterations, seed)

        summaries["separator_generation_time"][mode] = _summarize(separator_timings)
        summaries["prompt_assembly_time"][mode] = _summarize(prompt_timings)

        for iteration, elapsed_ms in enumerate(separator_timings, start=1):
            rows.append(
                {
                    "benchmark": "separator_generation_time",
                    "mode": mode,
                    "iteration": iteration,
                    "elapsed_ms": elapsed_ms,
                }
            )

        for iteration, elapsed_ms in enumerate(prompt_timings, start=1):
            rows.append(
                {
                    "benchmark": "prompt_assembly_time",
                    "mode": mode,
                    "iteration": iteration,
                    "elapsed_ms": elapsed_ms,
                }
            )

    _write_rows(rows, results_file)
    return summaries, rows


def main() -> None:
    summaries, _ = run_benchmark()
    print_summary_tables(summaries)
    print(f"Saved raw timings to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
