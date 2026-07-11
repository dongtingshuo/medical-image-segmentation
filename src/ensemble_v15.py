from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import boundary_f1_score, dice_score, iou_score, precision_score, recall_score, specificity_score


def _pad_to_multiple(images, multiple=32):
    height, width = images.shape[-2:]
    padded_height = int(np.ceil(height / multiple) * multiple)
    padded_width = int(np.ceil(width / multiple) * multiple)
    pad_height = padded_height - height
    pad_width = padded_width - width
    if pad_height == 0 and pad_width == 0:
        return images, (height, width)
    padded = F.pad(images, (0, pad_width, 0, pad_height), mode="replicate")
    return padded, (height, width)


def tta_probabilities(model, images, mode="none"):
    if mode not in {"none", "flip", "multiscale_flip"}:
        raise ValueError(f"Unsupported TTA mode: {mode}")
    height, width = images.shape[-2:]
    scales = [1.0] if mode != "multiscale_flip" else [0.875, 1.0, 1.125]
    flip_dimensions = [()] if mode == "none" else [(), (-1,), (-2,)]
    probabilities = []
    for scale in scales:
        if scale == 1.0:
            scaled = images
        else:
            scaled = F.interpolate(
                images,
                size=(max(32, int(round(height * scale))), max(32, int(round(width * scale)))),
                mode="bilinear",
                align_corners=False,
            )
        for dimensions in flip_dimensions:
            inputs = scaled if not dimensions else torch.flip(scaled, dims=dimensions)
            inputs, (scaled_height, scaled_width) = _pad_to_multiple(inputs)
            probability = torch.sigmoid(model(inputs))
            probability = probability[..., :scaled_height, :scaled_width]
            if dimensions:
                probability = torch.flip(probability, dims=dimensions)
            if probability.shape[-2:] != (height, width):
                probability = F.interpolate(probability, size=(height, width), mode="bilinear", align_corners=False)
            probabilities.append(probability)
    return torch.stack(probabilities).mean(0)


def macro_metrics(probabilities, targets, threshold=0.5, batch_size=16):
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    totals = {key: 0.0 for key in ["dice", "iou", "precision", "recall", "specificity", "boundary_f1"]}
    samples = 0
    for start in range(0, len(probabilities), int(batch_size)):
        probability = torch.from_numpy(np.asarray(probabilities[start : start + batch_size], dtype=np.float32))
        target = torch.from_numpy(np.asarray(targets[start : start + batch_size], dtype=np.float32))
        logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
        batch = len(probability)
        values = {
            "dice": dice_score(logits, target, threshold=threshold),
            "iou": iou_score(logits, target, threshold=threshold),
            "precision": precision_score(logits, target, threshold=threshold),
            "recall": recall_score(logits, target, threshold=threshold),
            "specificity": specificity_score(logits, target, threshold=threshold),
            "boundary_f1": boundary_f1_score(logits, target, threshold=threshold),
        }
        for key, value in values.items():
            totals[key] += float(value) * batch
        samples += batch
    metrics = {key: value / max(samples, 1) for key, value in totals.items()}
    metrics["composite"] = 0.75 * metrics["dice"] + 0.25 * metrics["boundary_f1"]
    metrics["samples"] = samples
    return metrics


def search_macro_threshold(probabilities, targets, start=0.20, stop=0.70, step=0.025):
    thresholds = np.arange(float(start), float(stop) + float(step) / 2.0, float(step))
    rows = []
    for threshold in thresholds:
        metrics = macro_metrics(probabilities, targets, threshold=float(threshold))
        rows.append({"threshold": float(round(threshold, 6)), **metrics})
    return max(rows, key=lambda row: (row["composite"], row["dice"])), rows


def average_probability_files(paths):
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one probability cache is required.")
    result = np.asarray(np.load(paths[0], mmap_mode="r"), dtype=np.float32).copy()
    for index, path in enumerate(paths[1:], start=2):
        value = np.load(path, mmap_mode="r")
        if value.shape != result.shape:
            raise ValueError(f"Probability cache shape mismatch: {path}")
        result += (np.asarray(value, dtype=np.float32) - result) / float(index)
    return result


def greedy_select_members(member_paths, targets, min_improvement=0.0005, max_members=5):
    if not member_paths:
        raise ValueError("No ensemble candidates were provided.")
    remaining = dict(member_paths)
    selected = []
    current = None
    history = []
    while remaining and len(selected) < int(max_members):
        best = None
        for name, path in sorted(remaining.items()):
            candidate_probability = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
            combined = candidate_probability.copy() if current is None else (
                current * len(selected) + candidate_probability
            ) / (len(selected) + 1)
            metrics, _ = search_macro_threshold(combined, targets)
            candidate = (metrics["composite"], metrics["dice"], name, combined, metrics)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        improvement = float("inf") if current is None else best[0] - history[-1]["composite"]
        if current is not None and improvement < float(min_improvement):
            break
        _, _, name, current, metrics = best
        selected.append(name)
        remaining.pop(name)
        history.append({"step": len(selected), "member": name, "improvement": improvement, **metrics})
    return selected, current, history


def postprocess_masks(probabilities, threshold, min_component_area=64, fill_holes=True):
    output = []
    for probability in probabilities:
        mask = (np.asarray(probability).squeeze() >= float(threshold)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= int(min_component_area):
                cleaned[labels == label] = 1
        if fill_holes:
            flood = cleaned.copy()
            padded = np.pad(flood, 1)
            cv2.floodFill(padded, None, (0, 0), 1)
            holes = 1 - padded[1:-1, 1:-1]
            cleaned = np.maximum(cleaned, holes)
        output.append(cleaned[None])
    return np.asarray(output, dtype=np.float32)


def write_decision(path, decision):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return path
