from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.analysis.error_analysis import analyze_segmentation_errors, summarize_error_records
from src.visualization import make_overlay


def _to_uint8_mask(mask):
    arr = np.asarray(mask).squeeze()
    if arr.dtype != np.uint8:
        arr = (arr > 0.5).astype(np.uint8) * 255
    return arr


def _read_resized_rgb(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _write_rgb(path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _to_uint8_mask(mask))


@torch.no_grad()
def collect_failure_case_records(model, dataloader, dataset, device, threshold=0.5):
    model.eval()
    records = []
    sample_index = 0
    for images, masks in dataloader:
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        true_masks = masks.detach().cpu().numpy()
        batch_size = probs.shape[0]
        for batch_index in range(batch_size):
            image_path, mask_path = dataset.pairs[sample_index]
            pred = probs[batch_index, 0] >= threshold
            true = true_masks[batch_index, 0] > 0.5
            analysis = analyze_segmentation_errors(pred.astype(np.uint8), true.astype(np.uint8))
            record = {
                "sample_index": sample_index,
                "image": image_path.name,
                "mask": mask_path.name,
                "threshold": float(threshold),
                "prediction_ratio": float(pred.mean()),
                "ground_truth_ratio": float(true.mean()),
                "pred_mask": pred.astype(np.uint8),
                **analysis,
            }
            records.append(record)
            sample_index += 1
    return records


def flatten_failure_record(record):
    flat = {
        "sample_index": record["sample_index"],
        "image": record["image"],
        "mask": record["mask"],
        "threshold": record["threshold"],
        "prediction_ratio": record["prediction_ratio"],
        "ground_truth_ratio": record["ground_truth_ratio"],
    }
    for key, value in record["metrics"].items():
        flat[key] = value
    for key, value in record["areas"].items():
        flat[f"area_{key}"] = value
    for key, value in record["error_flags"].items():
        flat[f"flag_{key}"] = value
    return flat


def select_failure_cases(records, top_k=16, sort_by="dice"):
    if not records:
        return []
    reverse = sort_by not in {"dice", "iou", "precision", "recall", "specificity"}
    if sort_by in records[0].get("metrics", {}):
        def key_fn(record):
            return float(record["metrics"].get(sort_by, 0.0))
    elif sort_by in records[0].get("areas", {}):
        def key_fn(record):
            return float(record["areas"].get(sort_by, 0.0))
    else:
        raise KeyError(f"Unsupported failure-case sort key: {sort_by}")
    return sorted(records, key=key_fn, reverse=reverse)[: int(top_k)]


def save_failure_case_visuals(records, dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, record in enumerate(records):
        image_path, _ = dataset.pairs[int(record["sample_index"])]
        true_mask = _to_uint8_mask(cv2.imread(str(dataset.pairs[int(record["sample_index"])][1]), cv2.IMREAD_GRAYSCALE))
        pred_mask = _to_uint8_mask(record["pred_mask"])
        size = pred_mask.shape[:2]
        image_rgb = _read_resized_rgb(image_path, size)
        true_resized = cv2.resize(true_mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        prefix = f"rank_{rank:02d}_idx_{record['sample_index']:05d}_{Path(record['image']).stem}"
        _write_rgb(output_dir / f"{prefix}_image.png", image_rgb)
        _write_mask(output_dir / f"{prefix}_true_mask.png", true_resized)
        _write_mask(output_dir / f"{prefix}_pred_mask.png", pred_mask)
        _write_rgb(output_dir / f"{prefix}_pred_overlay.png", make_overlay(image_rgb, pred_mask))
        _write_rgb(output_dir / f"{prefix}_true_overlay.png", make_overlay(image_rgb, true_resized))


def strip_large_arrays(records):
    stripped = []
    for record in records:
        item = dict(record)
        item.pop("pred_mask", None)
        stripped.append(item)
    return stripped


def write_failure_case_outputs(records, selected, output_dir, split="test", threshold=0.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_records = [flatten_failure_record(record) for record in records]
    csv_path = output_dir / "failure_cases.csv"
    json_path = output_dir / "failure_cases.json"
    md_path = output_dir / "failure_cases.md"
    if flat_records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_records[0].keys()))
            writer.writeheader()
            writer.writerows(flat_records)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(strip_large_arrays(records), indent=2), encoding="utf-8")
    summary = summarize_error_records(records)
    selected_rows = []
    for rank, record in enumerate(selected):
        flags = ", ".join([key for key, value in record["error_flags"].items() if value]) or "none"
        selected_rows.append(
            f"| {rank} | {record['sample_index']} | {record['image']} | {record['metrics']['dice']:.6f} | "
            f"{record['metrics']['iou']:.6f} | {record['metrics']['precision']:.6f} | "
            f"{record['metrics']['recall']:.6f} | {flags} |"
        )
    lines = [
        "# Failure Case Analysis",
        "",
        f"- Split: `{split}`",
        f"- Threshold: `{threshold:.3f}`",
        f"- Samples: `{summary['num_samples']}`",
        "",
        "## Mean Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Dice | {summary['mean_metrics']['dice']:.6f} |",
        f"| IoU | {summary['mean_metrics']['iou']:.6f} |",
        f"| Precision | {summary['mean_metrics']['precision']:.6f} |",
        f"| Recall | {summary['mean_metrics']['recall']:.6f} |",
        f"| Specificity | {summary['mean_metrics']['specificity']:.6f} |",
        "",
        "## Error Counts",
        "",
        "| Error Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in summary["error_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Worst Cases",
            "",
            "| Rank | Index | Image | Dice | IoU | Precision | Recall | Flags |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            *selected_rows,
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path, "summary": summary}
