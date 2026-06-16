import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.dataset import IMAGE_EXTENSIONS
from src.utils import data_path, load_config
from src.visualization import make_overlay


def list_images(directory):
    return sorted([p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def build_pairs(images_dir, masks_dir):
    images = list_images(images_dir)
    masks = list_images(masks_dir)
    mask_map = {p.stem: p for p in masks}
    pairs = []
    missing_masks = []
    for image_path in images:
        mask_path = mask_map.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))
    extra_masks = sorted(set(mask_map) - {p.stem for p in images})
    return images, masks, pairs, missing_masks, extra_masks


def read_image_and_mask(image_path, mask_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, mask


def inspect_masks(pairs):
    ratios = []
    warnings = []
    invalid_masks = []
    for _, mask_path in pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({0, 1, 255}):
            invalid_masks.append(mask_path.name)
        binary = mask > 127
        ratio = float(binary.mean())
        ratios.append(ratio)
        if ratio == 0.0:
            warnings.append(f"Mask is all black: {mask_path.name}")
        elif ratio == 1.0:
            warnings.append(f"Mask is all white: {mask_path.name}")
    return ratios, warnings, invalid_masks


def save_overlays(pairs, output_dir, sample_count=8, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected = rng.sample(pairs, k=min(sample_count, len(pairs)))
    saved_paths = []
    for idx, (image_path, mask_path) in enumerate(selected):
        image, mask = read_image_and_mask(image_path, mask_path)
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = make_overlay(image, mask > 127)
        canvas = np.concatenate([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), overlay], axis=1)
        out_path = output_dir / f"dataset_overlay_{idx:02d}_{image_path.stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        saved_paths.append(out_path)
    return saved_paths


def check_split(split_name, images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"{split_name} images directory does not exist: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"{split_name} masks directory does not exist: {masks_dir}")
    images, masks, pairs, missing_masks, extra_masks = build_pairs(images_dir, masks_dir)
    if len(images) != len(masks):
        raise ValueError(
            f"{split_name} image/mask count mismatch: images={len(images)}, masks={len(masks)}. "
            "Check file extensions and split directories."
        )
    if missing_masks or extra_masks:
        message = [f"{split_name} image/mask filename matching failed."]
        if missing_masks:
            message.append(f"Missing masks for images: {missing_masks[:10]}")
        if extra_masks:
            message.append(f"Extra masks without images: {extra_masks[:10]}")
        raise ValueError(" ".join(message))
    if not pairs:
        raise ValueError(f"{split_name} has no matched image/mask pairs.")
    ratios, warnings, invalid_masks = inspect_masks(pairs)
    return {
        "split": split_name,
        "images": images,
        "masks": masks,
        "pairs": pairs,
        "ratios": ratios,
        "warnings": warnings,
        "invalid_masks": invalid_masks,
    }


def write_report(report_path, train_info, val_info, saved_paths):
    all_ratios = train_info["ratios"] + val_info["ratios"]
    warnings = train_info["warnings"] + val_info["warnings"]
    invalid_masks = train_info["invalid_masks"] + val_info["invalid_masks"]
    lines = [
        "# Dataset Check Report",
        "",
        "## Summary",
        "",
        f"- Train images: {len(train_info['images'])}",
        f"- Train masks: {len(train_info['masks'])}",
        f"- Train matched pairs: {len(train_info['pairs'])}",
        f"- Val images: {len(val_info['images'])}",
        f"- Val masks: {len(val_info['masks'])}",
        f"- Val matched pairs: {len(val_info['pairs'])}",
        f"- Mean foreground ratio: {np.mean(all_ratios):.6f}",
        f"- Min foreground ratio: {np.min(all_ratios):.6f}",
        f"- Max foreground ratio: {np.max(all_ratios):.6f}",
        "",
        "## Binary Mask Check",
        "",
        f"- Invalid binary masks: {len(invalid_masks)}",
    ]
    if invalid_masks:
        lines.append(f"- Examples: {invalid_masks[:20]}")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend([f"- {item}" for item in warnings[:50]])
    else:
        lines.append("- No all-black or all-white masks found.")
    lines.extend(["", "## Saved Overlay Samples", ""])
    lines.extend([f"- {path}" for path in saved_paths])
    lines.extend(
        [
            "",
            "## Pass Criteria",
            "",
            "- Image and mask counts are equal for each split.",
            "- Image and mask filename stems match.",
            "- Masks are binary or can be safely thresholded.",
            "- Overlay samples align with visible lesion regions.",
        ]
    )
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / "sanity_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_info = check_split("train", data_path(config, "train_images_dir"), data_path(config, "train_masks_dir"))
    val_info = check_split("val", data_path(config, "val_images_dir"), data_path(config, "val_masks_dir"))
    saved_paths = save_overlays(train_info["pairs"], output_dir, sample_count=8, seed=config.get("seed", 42))
    report_path = output_dir / "dataset_check_report.md"
    write_report(report_path, train_info, val_info, saved_paths)

    print(f"Train pairs: {len(train_info['pairs'])}")
    print(f"Val pairs: {len(val_info['pairs'])}")
    print(f"Saved overlays to: {output_dir}")
    print(f"Saved report to: {report_path}")
    if train_info["warnings"] or val_info["warnings"]:
        print("Warning: all-black or all-white masks were found. Inspect dataset_check_report.md before training.")
    if train_info["invalid_masks"] or val_info["invalid_masks"]:
        print("Warning: non-binary mask values were found. They will be thresholded during training.")


if __name__ == "__main__":
    main()
