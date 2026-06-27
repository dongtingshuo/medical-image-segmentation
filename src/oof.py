from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.metrics import boundary_f1_score, dice_score


def probability_logits(probability):
    probability = torch.as_tensor(probability).float().clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probability)


def sample_scores(probability, target, threshold=0.5):
    probability = torch.as_tensor(probability).float()
    target = torch.as_tensor(target).float()
    if probability.ndim == 2:
        probability = probability[None, None]
    if target.ndim == 2:
        target = target[None, None]
    logits = probability_logits(probability)
    return {
        "dice": float(dice_score(logits, target, threshold=threshold)),
        "boundary_f1": float(boundary_f1_score(logits, target, threshold=threshold)),
    }


def difficulty_weight(dice, boundary_f1, low_contrast):
    difficulty = 0.55 * (1.0 - float(dice)) + 0.30 * (1.0 - float(boundary_f1)) + 0.15 * float(low_contrast)
    return float(np.clip(1.0 + 2.0 * difficulty, 1.0, 3.0)), float(difficulty)


def restore_probability(probability, original_shape, resize_mode="letterbox"):
    probability = np.asarray(probability, dtype=np.float32)
    original_height, original_width = map(int, original_shape[:2])
    transformed_height, transformed_width = probability.shape[-2:]
    if resize_mode == "stretch":
        return cv2.resize(probability, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    if resize_mode != "letterbox":
        raise ValueError(f"Unsupported resize mode: {resize_mode}")
    scale = min(transformed_height / original_height, transformed_width / original_width)
    resized_height = max(1, int(round(original_height * scale)))
    resized_width = max(1, int(round(original_width * scale)))
    top = max(0, (transformed_height - resized_height) // 2)
    left = max(0, (transformed_width - resized_width) // 2)
    cropped = probability[top : top + resized_height, left : left + resized_width]
    return cv2.resize(cropped, (original_width, original_height), interpolation=cv2.INTER_LINEAR)


def resize_target(mask, output_shape, resize_mode="letterbox"):
    output_height, output_width = map(int, output_shape[:2])
    mask = np.asarray(mask, dtype=np.float32)
    if resize_mode == "stretch":
        return cv2.resize(mask, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    if resize_mode != "letterbox":
        raise ValueError(f"Unsupported resize mode: {resize_mode}")
    scale = min(output_height / mask.shape[0], output_width / mask.shape[1])
    resized_height = max(1, int(round(mask.shape[0] * scale)))
    resized_width = max(1, int(round(mask.shape[1] * scale)))
    resized = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    output = np.zeros((output_height, output_width), dtype=np.float32)
    top = (output_height - resized_height) // 2
    left = (output_width - resized_width) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


def write_soft_mask(path, probability):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.rint(np.clip(probability, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise OSError(f"Failed to write OOF soft mask: {path}")
    return path


def validate_oof_coverage(folds, predictions, architectures):
    expected = {stem for fold in folds for stem in fold["val_ids"]}
    fold_counts = {stem: 0 for stem in expected}
    for fold in folds:
        for stem in fold["val_ids"]:
            fold_counts[stem] += 1
    invalid_folds = sorted(stem for stem, count in fold_counts.items() if count != 1)
    if invalid_folds:
        raise ValueError(f"OOF validation partition is invalid for: {invalid_folds[:10]}")
    missing = []
    for architecture in architectures:
        available = set(predictions.get(architecture, {}))
        missing.extend(f"{architecture}:{stem}" for stem in sorted(expected - available))
    if missing:
        raise ValueError(f"OOF predictions are incomplete: {missing[:10]}")
    return sorted(expected)


def write_oof_outputs(predictions, folds, manifest_rows, images_dir, masks_dir, output_root, resize_mode="letterbox"):
    architectures = sorted(predictions)
    stems = validate_oof_coverage(folds, predictions, architectures)
    images_dir, masks_dir, output_root = Path(images_dir), Path(masks_dir), Path(output_root)
    manifest = {row["stem"]: row for row in manifest_rows if row.get("stem")}
    image_paths = {path.stem: path for path in images_dir.iterdir() if path.is_file()}
    mask_paths = {path.stem: path for path in masks_dir.iterdir() if path.is_file()}
    missing_manifest = sorted(set(stems) - set(manifest))
    missing_images = sorted(set(stems) - set(image_paths))
    missing_masks = sorted(set(stems) - set(mask_paths))
    if missing_manifest or missing_images or missing_masks:
        raise ValueError(
            "OOF source coverage mismatch: "
            f"manifest={missing_manifest[:10]}, images={missing_images[:10]}, masks={missing_masks[:10]}"
        )
    rows = []
    soft_dir = output_root / "soft_masks"
    for stem in stems:
        probabilities = [np.asarray(predictions[architecture][stem], dtype=np.float32) for architecture in architectures]
        mean_probability = np.mean(probabilities, axis=0, dtype=np.float32)
        image = cv2.imread(str(image_paths[stem]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_paths[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Failed to read OOF source pair: {stem}")
        restored = restore_probability(mean_probability, image.shape[:2], resize_mode=resize_mode)
        write_soft_mask(soft_dir / f"{stem}.png", restored)
        transformed_target = resize_target(
            (mask > 127).astype(np.float32), mean_probability.shape, resize_mode=resize_mode
        )
        scores = sample_scores(mean_probability, transformed_target)
        contrast_bin = int(manifest[stem].get("contrast_bin", 1))
        low_contrast = {0: 1.0, 1: 0.5, 2: 0.0}.get(contrast_bin, 0.5)
        weight, difficulty = difficulty_weight(scores["dice"], scores["boundary_f1"], low_contrast)
        rows.append(
            {
                "stem": stem,
                "dice": scores["dice"],
                "boundary_f1": scores["boundary_f1"],
                "low_contrast": low_contrast,
                "difficulty": difficulty,
                "weight": weight,
                "architectures": ";".join(architectures),
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    weights_path = output_root / "hard_example_weights.csv"
    with weights_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    coverage = {
        "architectures": architectures,
        "samples": len(stems),
        "folds": len(folds),
        "complete": True,
    }
    (output_root / "oof_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return {"soft_masks_dir": soft_dir, "weights": weights_path, "coverage": coverage, "rows": rows}
