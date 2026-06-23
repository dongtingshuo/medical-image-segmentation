import csv

import pytest

from src.experiment_suite import read_single_row_csv, summarize_seed_metrics, write_summary_markdown


def test_summarize_seed_metrics_uses_sample_standard_deviation():
    rows = [
        {"seed": 1, "split": "test", "dice": "0.8", "iou": "0.7"},
        {"seed": 2, "split": "test", "dice": "0.9", "iou": "0.8"},
        {"seed": 3, "split": "test", "dice": "1.0", "iou": "0.9"},
    ]
    summary = summarize_seed_metrics(rows, metrics=("dice", "iou"))
    dice = next(item for item in summary if item["metric"] == "dice")
    assert dice["mean"] == pytest.approx(0.9)
    assert dice["std"] == pytest.approx(0.1)
    assert dice["runs"] == 3


def test_single_row_reader_and_markdown_writer(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "dice"])
        writer.writeheader()
        writer.writerow({"split": "val", "dice": "0.8"})
    assert read_single_row_csv(csv_path)["dice"] == "0.8"

    report = tmp_path / "summary.md"
    write_summary_markdown(
        report,
        [{"split": "val"}],
        [{"split": "val", "metric": "dice", "runs": 1, "mean": 0.8, "std": 0, "min": 0.8, "max": 0.8}],
        best_seed=42,
    )
    assert "Best seed by validation Dice: `42`" in report.read_text(encoding="utf-8")
