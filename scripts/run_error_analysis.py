import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from src.analysis.error_analysis import (
    analyze_segmentation_errors,
    summarize_error_records,
    write_error_analysis_report,
)


def _ensure_data(data_dir):
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    if not masks_dir.exists() or not list(masks_dir.glob("*.png")):
        create_toy_segmentation_data(data_dir)
    return images_dir, masks_dir


def _mock_predictions(mask):
    kernel = np.ones((7, 7), dtype=np.uint8)
    return [
        ("boundary_like_mismatch", np.roll(mask, shift=5, axis=1)),
        ("over_segmentation", cv2.dilate(mask, kernel, iterations=2)),
        ("under_segmentation", cv2.erode(mask, kernel, iterations=2)),
        ("empty_prediction", np.zeros_like(mask)),
        ("full_prediction", np.ones_like(mask) * 255),
    ]


def run_error_analysis(data_dir="examples/toy_segmentation_demo", output_dir="outputs/analysis"):
    _, masks_dir = _ensure_data(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = sorted(masks_dir.glob("*.png"))[0]
    true_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    records = []
    for label, pred in _mock_predictions(true_mask):
        record = analyze_segmentation_errors(pred, true_mask)
        record["sample"] = label
        records.append(record)
    empty_true = np.zeros_like(true_mask)
    records.append({"sample": "empty_ground_truth", **analyze_segmentation_errors(true_mask, empty_true)})
    summary = summarize_error_records(records)
    payload = {"summary": summary, "records": records}
    json_path = output_dir / "error_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = output_dir / "error_analysis.md"
    write_error_analysis_report(summary, records, report_path)
    return output_dir, json_path, report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run toy segmentation error analysis.")
    parser.add_argument("--data-dir", default="examples/toy_segmentation_demo")
    parser.add_argument("--output-dir", default="outputs/analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir, json_path, report_path = run_error_analysis(args.data_dir, args.output_dir)
    print(f"Error analysis outputs saved to: {output_dir}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
