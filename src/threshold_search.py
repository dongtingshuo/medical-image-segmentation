from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

THRESHOLD_METRICS = ("dice", "iou", "precision", "recall", "specificity")


def parse_thresholds(value=None, start=0.3, stop=0.7, step=0.05):
    if value:
        thresholds = [float(item) for item in str(value).replace(" ", "").split(",") if item]
    else:
        count = int(round((float(stop) - float(start)) / float(step))) + 1
        thresholds = [float(start) + index * float(step) for index in range(count)]
    thresholds = [round(float(threshold), 6) for threshold in thresholds]
    invalid = [threshold for threshold in thresholds if threshold < 0.0 or threshold > 1.0]
    if invalid:
        raise ValueError(f"Thresholds must be between 0 and 1, got {invalid}")
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def compute_threshold_metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


def summarize_threshold_predictions(probability_batches, mask_batches, thresholds):
    rows = []
    thresholds = parse_thresholds(",".join(str(threshold) for threshold in thresholds))
    for threshold in thresholds:
        tp = fp = fn = tn = 0.0
        pixels = 0
        samples = 0
        if len(probability_batches) != len(mask_batches):
            raise ValueError("Probability and mask batch counts do not match.")
        for probabilities, masks in zip(probability_batches, mask_batches):
            probs = torch.as_tensor(probabilities).detach().cpu()
            true = torch.as_tensor(masks).detach().cpu() > 0.5
            pred = probs >= threshold
            tp += float((pred & true).sum().item())
            fp += float((pred & ~true).sum().item())
            fn += float((~pred & true).sum().item())
            tn += float((~pred & ~true).sum().item())
            pixels += int(true.numel())
            samples += int(true.shape[0]) if true.ndim >= 3 else 1
        row = {
            "threshold": threshold,
            "samples": samples,
            "pixels": pixels,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        row.update(compute_threshold_metrics_from_counts(tp, fp, fn, tn))
        rows.append(row)
    return rows


@torch.no_grad()
def collect_probability_batches(model, dataloader, device):
    model.eval()
    probability_batches = []
    mask_batches = []
    for images, masks in dataloader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits).detach().cpu()
        probability_batches.append(probabilities)
        mask_batches.append(masks.detach().cpu())
    return probability_batches, mask_batches


def best_threshold(rows, metric="dice"):
    if not rows:
        raise ValueError("No threshold rows were provided.")
    if metric not in rows[0]:
        raise KeyError(f"Metric `{metric}` is not present in threshold rows.")
    return max(rows, key=lambda row: (float(row[metric]), float(row["threshold"])))


def write_threshold_search_outputs(rows, output_dir, best_metric="dice", split="val"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_search.csv"
    json_path = output_dir / "threshold_search.json"
    md_path = output_dir / "threshold_search.md"
    if not rows:
        raise ValueError("No threshold rows to write.")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = best_threshold(rows, best_metric)
    lines = [
        "# Threshold Search Report",
        "",
        f"- Split: `{split}`",
        f"- Best metric: `{best_metric}`",
        f"- Best threshold: `{best['threshold']:.3f}`",
        f"- Best Dice: `{best['dice']:.6f}`",
        f"- Best IoU: `{best['iou']:.6f}`",
        f"- Best Precision: `{best['precision']:.6f}`",
        f"- Best Recall: `{best['recall']:.6f}`",
        f"- Best Specificity: `{best['specificity']:.6f}`",
        "",
        "| Threshold | Dice | IoU | Precision | Recall | Specificity |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.3f} | {row['dice']:.6f} | {row['iou']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['specificity']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path, "best": best}
