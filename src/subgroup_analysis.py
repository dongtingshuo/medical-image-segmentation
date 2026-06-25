from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np


def image_contrast(image_rgb):
    gray = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2GRAY)
    return float(gray.std())


def lesion_size_group(foreground_ratio):
    ratio = float(foreground_ratio)
    if ratio < 0.05:
        return "small"
    if ratio < 0.20:
        return "medium"
    return "large"


def tertile_thresholds(values):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("Cannot compute tertiles from empty values.")
    low = values[int((len(values) - 1) / 3)]
    high = values[int(2 * (len(values) - 1) / 3)]
    return low, high


def tertile_group(value, low, high, labels=("low", "medium", "high")):
    value = float(value)
    if value <= low:
        return labels[0]
    if value <= high:
        return labels[1]
    return labels[2]


def attach_subgroups(records):
    if not records:
        return []
    contrast_values = [float(record["image_contrast"]) for record in records]
    lesion_values = [float(record["ground_truth_ratio"]) for record in records]
    contrast_low, contrast_high = tertile_thresholds(contrast_values)
    lesion_low, lesion_high = tertile_thresholds(lesion_values)
    enriched = []
    for record in records:
        item = dict(record)
        item["lesion_size_group"] = lesion_size_group(item["ground_truth_ratio"])
        item["lesion_ratio_tertile"] = tertile_group(item["ground_truth_ratio"], lesion_low, lesion_high)
        item["contrast_tertile"] = tertile_group(item["image_contrast"], contrast_low, contrast_high)
        enriched.append(item)
    return enriched


def summarize_subgroups(records, group_fields=("lesion_size_group", "lesion_ratio_tertile", "contrast_tertile")):
    metrics = ("dice", "iou", "precision", "recall", "specificity")
    rows = []
    for group_field in group_fields:
        grouped = {}
        for record in records:
            grouped.setdefault(record[group_field], []).append(record)
        for group_value, group_records in sorted(grouped.items()):
            row = {"group_field": group_field, "group": group_value, "samples": len(group_records)}
            for metric in metrics:
                values = [float(record[metric]) for record in group_records]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            rows.append(row)
    return rows


def write_subgroup_outputs(records, summary_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "subgroup_per_sample.csv"
    summary_path = output_dir / "subgroup_summary.csv"
    json_path = output_dir / "subgroup_summary.json"
    md_path = output_dir / "subgroup_summary.md"
    if records:
        with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    else:
        per_sample_path.write_text("", encoding="utf-8")
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# Subgroup Analysis",
        "",
        "| Group Field | Group | Samples | Dice | IoU | Precision | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group_field']} | {row['group']} | {row['samples']} | "
            f"{row['dice_mean']:.6f} | {row['iou_mean']:.6f} | "
            f"{row['precision_mean']:.6f} | {row['recall_mean']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "per_sample": per_sample_path,
        "summary": summary_path,
        "json": json_path,
        "markdown": md_path,
    }
