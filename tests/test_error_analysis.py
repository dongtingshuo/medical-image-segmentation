import numpy as np

from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from scripts.run_error_analysis import run_error_analysis
from src.analysis.error_analysis import analyze_segmentation_errors
from src.analysis.failure_cases import select_failure_cases, write_failure_case_outputs


def _square(size=32, start=10, end=20):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[start:end, start:end] = 255
    return mask


def test_error_analysis_handles_common_boundary_cases():
    true = _square()
    empty = np.zeros_like(true)
    full = np.ones_like(true) * 255
    small_pred = _square(start=13, end=17)
    shifted = np.roll(true, shift=4, axis=1)

    assert analyze_segmentation_errors(empty, true)["error_flags"]["empty_prediction"]
    assert analyze_segmentation_errors(full, true)["error_flags"]["over_segmentation"]
    assert analyze_segmentation_errors(small_pred, true)["error_flags"]["under_segmentation"]
    assert analyze_segmentation_errors(shifted, true)["error_flags"]["boundary_error"]
    assert analyze_segmentation_errors(true, empty)["error_flags"]["empty_ground_truth"]


def test_run_error_analysis_outputs_json_and_bilingual_report(tmp_path):
    data_dir = tmp_path / "toy"
    create_toy_segmentation_data(data_dir, num_samples=2, image_size=64)
    output_dir, json_path, report_path = run_error_analysis(data_dir=data_dir, output_dir=tmp_path / "analysis")
    assert output_dir.exists()
    assert json_path.exists()
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Error Analysis Overview / 错误分析概述" in text
    assert "Medical Disclaimer / 医学免责声明" in text


def test_failure_case_report_ranks_worst_dice(tmp_path):
    true = _square()
    good = analyze_segmentation_errors(true, true)
    bad = analyze_segmentation_errors(np.zeros_like(true), true)
    records = [
        {"sample_index": 0, "image": "good.png", "mask": "good.png", "threshold": 0.5, "prediction_ratio": 0.1, "ground_truth_ratio": 0.1, **good},
        {"sample_index": 1, "image": "bad.png", "mask": "bad.png", "threshold": 0.5, "prediction_ratio": 0.0, "ground_truth_ratio": 0.1, **bad},
    ]
    selected = select_failure_cases(records, top_k=1, sort_by="dice")
    assert selected[0]["image"] == "bad.png"

    outputs = write_failure_case_outputs(records, selected, tmp_path, split="test", threshold=0.5)
    text = outputs["markdown"].read_text(encoding="utf-8")
    assert "Failure Case Analysis" in text
    assert "bad.png" in text
