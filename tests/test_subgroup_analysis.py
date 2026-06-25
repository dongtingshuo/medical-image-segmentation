import numpy as np
import pytest

from src.subgroup_analysis import (
    attach_subgroups,
    image_contrast,
    lesion_size_group,
    summarize_subgroups,
    tertile_group,
    tertile_thresholds,
)


def test_lesion_size_group_thresholds():
    assert lesion_size_group(0.01) == "small"
    assert lesion_size_group(0.10) == "medium"
    assert lesion_size_group(0.30) == "large"


def test_attach_subgroups_and_summary():
    records = [
        {"ground_truth_ratio": 0.01, "image_contrast": 5.0, "dice": 0.5, "iou": 0.4, "precision": 0.6, "recall": 0.7, "specificity": 0.9},
        {"ground_truth_ratio": 0.10, "image_contrast": 10.0, "dice": 0.7, "iou": 0.6, "precision": 0.8, "recall": 0.8, "specificity": 0.95},
        {"ground_truth_ratio": 0.30, "image_contrast": 20.0, "dice": 0.9, "iou": 0.8, "precision": 0.9, "recall": 0.9, "specificity": 0.99},
    ]
    enriched = attach_subgroups(records)
    assert {record["lesion_size_group"] for record in enriched} == {"small", "medium", "large"}

    summary = summarize_subgroups(enriched, group_fields=("lesion_size_group",))
    large = next(row for row in summary if row["group"] == "large")
    assert large["dice_mean"] == pytest.approx(0.9)


def test_tertiles_and_image_contrast():
    low, high = tertile_thresholds([1, 2, 3, 4, 5, 6])
    assert tertile_group(low, low, high) == "low"
    assert tertile_group(high, low, high) in {"medium", "high"}
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255
    assert image_contrast(image) > 0
