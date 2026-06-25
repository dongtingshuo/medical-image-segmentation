import pytest

from src.statistical_analysis import bootstrap_mean_difference, group_metric_summary, mean_std_ci


def test_mean_std_ci_for_multiple_values():
    result = mean_std_ci([0.8, 0.9, 1.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(0.9)
    assert result["std"] == pytest.approx(0.1)
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_bootstrap_mean_difference_is_deterministic():
    result = bootstrap_mean_difference([0.7, 0.8, 0.9], [0.8, 0.9, 1.0], iterations=100, seed=123)
    repeat = bootstrap_mean_difference([0.7, 0.8, 0.9], [0.8, 0.9, 1.0], iterations=100, seed=123)
    assert result == repeat
    assert result["observed_difference"] == pytest.approx(0.1)
    assert result["paired"] is True


def test_group_metric_summary_groups_by_encoder_and_split():
    rows = [
        {"encoder": "a", "split": "val", "dice": "0.8", "iou": "0.7"},
        {"encoder": "a", "split": "val", "dice": "0.9", "iou": "0.8"},
        {"encoder": "b", "split": "val", "dice": "0.7", "iou": "0.6"},
    ]
    summary = group_metric_summary(rows, ["encoder", "split"], ["dice", "iou"])
    dice_a = next(row for row in summary if row["encoder"] == "a" and row["metric"] == "dice")
    assert dice_a["n"] == 2
    assert dice_a["mean"] == pytest.approx(0.85)
