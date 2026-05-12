import csv

import latency_benchmark


def test_run_benchmark_writes_expected_rows(tmp_path, capsys):
    results_file = tmp_path / "latency_benchmark.csv"

    summaries, rows = latency_benchmark.run_benchmark(
        iterations=5,
        seed=1337,
        results_file=results_file,
    )
    latency_benchmark.print_summary_tables(summaries)

    captured = capsys.readouterr().out

    assert results_file.exists()
    assert len(rows) == 20
    assert set(summaries) == {
        "separator_generation_time",
        "prompt_assembly_time",
    }

    with results_file.open("r", newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 20
    assert {row["benchmark"] for row in csv_rows} == set(summaries)
    assert {row["mode"] for row in csv_rows} == {"static", "dynamic"}
    assert "separator_generation_time" in captured
    assert "prompt_assembly_time" in captured
    assert "mode | mean_ms | std_ms | min_ms | max_ms" in captured

    for row in csv_rows:
        assert int(row["iteration"]) >= 1
        assert float(row["elapsed_ms"]) >= 0.0
