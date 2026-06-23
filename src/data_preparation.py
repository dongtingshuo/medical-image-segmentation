import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_SUFFIX = re.compile(r"(?:_segmentation|_mask|_label|_ground_?truth)$", re.IGNORECASE)


def normalized_sample_id(path):
    return MASK_SUFFIX.sub("", Path(path).stem).lower()


def infer_split(relative_path):
    tokens = [token.lower() for token in Path(relative_path).parts]
    if any("test" in token for token in tokens):
        return "test"
    if any("validation" in token or token == "val" or token.startswith("val_") for token in tokens):
        return "val"
    if any("train" in token for token in tokens):
        return "train"
    return "unknown"


def is_mask_path(relative_path):
    path = Path(relative_path)
    parent_tokens = [token.lower() for token in path.parts[:-1]]
    mask_parent = any(
        any(marker in token for marker in ("mask", "groundtruth", "ground_truth", "segmentation"))
        for token in parent_tokens
    )
    return bool(mask_parent or MASK_SUFFIX.search(path.stem))


def scan_pairs(source_root):
    source_root = Path(source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {source_root}")
    images = defaultdict(dict)
    masks = defaultdict(dict)
    duplicates = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(source_root)
        split = infer_split(relative)
        sample_id = normalized_sample_id(path)
        target = masks if is_mask_path(relative) else images
        if sample_id in target[split]:
            duplicates.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "kept": str(target[split][sample_id]),
                    "ignored": str(path),
                }
            )
            continue
        target[split][sample_id] = path

    pairs = {}
    for split in sorted(set(images) | set(masks)):
        common = sorted(set(images[split]) & set(masks[split]))
        pairs[split] = [(sample_id, images[split][sample_id], masks[split][sample_id]) for sample_id in common]
    report = {
        "source_root": str(source_root),
        "image_counts": {split: len(values) for split, values in images.items()},
        "mask_counts": {split: len(values) for split, values in masks.items()},
        "pair_counts": {split: len(values) for split, values in pairs.items()},
        "duplicate_count": len(duplicates),
        "duplicates": duplicates[:100],
    }
    return pairs, report


def _write_pair(image_path, mask_path, output_images, output_masks, sample_id, image_size):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_images / f"{sample_id}.jpg"), image):
        raise OSError(f"Failed to write prepared image: {sample_id}")
    if not cv2.imwrite(str(output_masks / f"{sample_id}.png"), mask):
        raise OSError(f"Failed to write prepared mask: {sample_id}")


def prepare_internal_splits(source_root, output_root, image_size=384):
    pairs, report = scan_pairs(source_root)
    required = ("train", "val", "test")
    missing = [split for split in required if not pairs.get(split)]
    if missing:
        raise ValueError(
            f"Internal dataset is missing paired splits: {missing}. Detected pair counts: {report['pair_counts']}"
        )
    output_root = Path(output_root)
    for split in required:
        for sample_id, image_path, mask_path in pairs[split]:
            _write_pair(
                image_path,
                mask_path,
                output_root / split / "images",
                output_root / split / "masks",
                sample_id,
                int(image_size),
            )
    report.update(
        {
            "kind": "internal",
            "prepared_root": str(output_root),
            "prepared_splits": {split: len(pairs[split]) for split in required},
            "image_size": int(image_size),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def collect_sample_ids(source_root):
    pairs, _ = scan_pairs(source_root)
    return {sample_id for split_pairs in pairs.values() for sample_id, _, _ in split_pairs}


def prepare_external_split(
    source_root,
    output_root,
    excluded_ids=None,
    image_size=384,
    preferred_splits=("test", "val", "train", "unknown"),
):
    pairs, report = scan_pairs(source_root)
    excluded_ids = {str(sample_id).lower() for sample_id in (excluded_ids or set())}
    selected_split = next((split for split in preferred_splits if pairs.get(split)), None)
    if selected_split is None:
        raise ValueError(f"External dataset contains no image/mask pairs. Detected: {report['pair_counts']}")
    selected_pairs = pairs[selected_split]
    overlap_ids = sorted(sample_id for sample_id, _, _ in selected_pairs if sample_id in excluded_ids)
    selected_pairs = [pair for pair in selected_pairs if pair[0] not in excluded_ids]
    if not selected_pairs:
        raise ValueError("No external pairs remain after excluding IDs present in the internal dataset.")

    output_root = Path(output_root)
    for sample_id, image_path, mask_path in selected_pairs:
        _write_pair(
            image_path,
            mask_path,
            output_root / "images",
            output_root / "masks",
            sample_id,
            int(image_size),
        )
    report.update(
        {
            "kind": "external",
            "prepared_root": str(output_root),
            "selected_source_split": selected_split,
            "selected_before_exclusion": len(pairs[selected_split]),
            "excluded_overlap_count": len(overlap_ids),
            "excluded_overlap_ids": overlap_ids,
            "prepared_pairs": len(selected_pairs),
            "image_size": int(image_size),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
