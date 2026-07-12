"""Lightweight Kaggle preflight for v1.6 mounted datasets and GPU runtime."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multisource_data import discover_pairs  # noqa: E402
from src.v16 import materialized_stem, read_ham_metadata  # noqa: E402


def inspect_pairs(label, images_dir, masks_dir, require_metadata=False):
    pairs = discover_pairs(images_dir, masks_dir)
    stem, image_path, mask_path = pairs[0]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"{label} first pair has invalid geometry: {image_path}, {mask_path}")
    return {"pairs": len(pairs), "first_stem": stem, "image_shape": list(image.shape), "mask_shape": list(mask.shape)}


def inspect_ham_error_stem(images_dir, masks_dir, metadata, stem="HAM_0000008"):
    matches = [
        {
            "stem": pair_stem,
            "image": str(image_path.relative_to(images_dir)),
            "mask": str(mask_path.relative_to(masks_dir)),
        }
        for pair_stem, image_path, mask_path in discover_pairs(images_dir, masks_dir)
        if pair_stem == stem
    ]
    return {
        "requested_stem": stem,
        "discovered_pairs": matches,
        "metadata_present": stem in metadata,
        "expected_materialized_stem": materialized_stem("ham10000", stem),
    }


def main():
    parser = argparse.ArgumentParser(description="Check v1.6 Kaggle inputs without starting long training.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--isic16-images", required=True)
    parser.add_argument("--isic16-masks", required=True)
    parser.add_argument("--ph2-images", required=True)
    parser.add_argument("--ph2-masks", required=True)
    parser.add_argument("--ham-images", required=True)
    parser.add_argument("--ham-masks", required=True)
    parser.add_argument("--ham-metadata", required=True)
    args = parser.parse_args()

    metadata = read_ham_metadata(args.ham_metadata)
    report = {
        "isic16": inspect_pairs("isic16", args.isic16_images, args.isic16_masks),
        "ph2": inspect_pairs("ph2", args.ph2_images, args.ph2_masks),
        "ham10000": inspect_pairs("ham10000", args.ham_images, args.ham_masks),
        "ham_metadata_images": len(metadata),
        "ham_error_stem": inspect_ham_error_stem(args.ham_images, args.ham_masks, metadata),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "wandb": "disabled",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
