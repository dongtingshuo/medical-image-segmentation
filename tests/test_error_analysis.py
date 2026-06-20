import numpy as np

from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from scripts.run_error_analysis import run_error_analysis
from src.analysis.error_analysis import analyze_segmentation_errors


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
