from __future__ import annotations

import csv
import hashlib
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = ("_segmentation", "_lesion", "_mask", "_label", "_groundtruth", "_ground_truth")
MASK_PARENT_MARKERS = ("mask", "groundtruth", "ground_truth", "segmentation")


def normalized_stem(path):
    stem = Path(path).stem
    lowered = stem.lower()
    for suffix in MASK_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _looks_like_mask(path, root=None):
    path = Path(path)
    lowered_stem = path.stem.lower()
    parent_parts = path.relative_to(root).parts[:-1] if root is not None else path.parts[:-1]
    if any(lowered_stem.endswith(suffix) for suffix in MASK_SUFFIXES):
        return True
    for part in parent_parts:
        lowered_part = part.lower()
        if any(marker in lowered_part for marker in MASK_PARENT_MARKERS):
            return True
        if "lesion" in lowered_part and not any(marker in lowered_part for marker in ("image", "photo", "dermoscopic")):
            return True
    return False


def _index_candidates(candidates, label):
    indexed = {}
    duplicates = []
    for path in candidates:
        stem = normalized_stem(path)
        if stem in indexed:
            duplicates.append((stem, indexed[stem], path))
        else:
            indexed[stem] = path
    if duplicates:
        stem, first, second = duplicates[0]
        raise ValueError(f"Duplicate normalized {label} stem `{stem}`: {first}, {second}")
    return indexed


def discover_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Source directories do not exist: images={images_dir}, masks={masks_dir}")
    shared_root = images_dir.resolve() == masks_dir.resolve()
    image_candidates = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (not shared_root or not _looks_like_mask(path, images_dir))
    ]
    mask_candidates = [
        path
        for path in sorted(masks_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (not shared_root or _looks_like_mask(path, masks_dir))
    ]
    images = _index_candidates(image_candidates, "image")
    masks = _index_candidates(mask_candidates, "mask")
    common = sorted(set(images) & set(masks))
    if not common:
        raise ValueError(f"No image/mask pairs found under {images_dir} and {masks_dir}")
    return [(stem, images[stem], masks[stem]) for stem in common]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image, hash_size=8, high_frequency_factor=4):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    size = int(hash_size) * int(high_frequency_factor)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:hash_size, :hash_size]
    median = float(np.median(dct[1:, 1:]))
    bits = (dct > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left, right):
    return bin(int(left) ^ int(right)).count("1")


def structural_similarity(left, right, size=128):
    left = cv2.resize(left, (size, size), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (size, size), interpolation=cv2.INTER_AREA)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float64)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mean_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mean_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    var_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mean_left * mean_left
    var_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mean_right * mean_right
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mean_left * mean_right
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2 * mean_left * mean_right + c1) * (2 * covariance + c2)
    denominator = (mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sample_features(image_path, mask_path):
    image = read_rgb(image_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return {
        "sha256": sha256_file(image_path),
        "phash": perceptual_hash(image),
        "contrast": float(gray.std()),
        "lesion_ratio": float((mask > 127).mean()),
        "path": str(image_path),
    }


def _find_duplicate(features, references, phash_distance=4, ssim_threshold=0.95):
    for reference in references:
        if features["sha256"] == reference["sha256"]:
            return reference, "sha256", 1.0
    candidate_image = None
    for reference in references:
        if hamming_distance(features["phash"], reference["phash"]) <= int(phash_distance):
            if candidate_image is None:
                candidate_image = read_rgb(features["path"])
            score = structural_similarity(candidate_image, read_rgb(reference["path"]))
            if score >= float(ssim_threshold):
                return reference, "phash_ssim", score
    return None, "", None


def _safe_stem(source, stem):
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in stem)
    return f"{source}__{normalized}"


def prepare_multisource_dataset(sources, benchmarks, output_root, phash_distance=4, ssim_threshold=0.95):
    output_root = Path(output_root)
    images_out = output_root / "images"
    masks_out = output_root / "masks"
    if images_out.exists():
        shutil.rmtree(images_out)
    if masks_out.exists():
        shutil.rmtree(masks_out)
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    benchmark_references = []
    for source, images_dir, masks_dir in benchmarks:
        for stem, image_path, mask_path in discover_pairs(images_dir, masks_dir):
            features = sample_features(image_path, mask_path)
            benchmark_references.append({**features, "source": source, "stem": stem, "is_benchmark": True})

    accepted_references = []
    rows = []
    for source, images_dir, masks_dir in sources:
        for stem, image_path, mask_path in discover_pairs(images_dir, masks_dir):
            features = sample_features(image_path, mask_path)
            duplicate, reason, score = _find_duplicate(
                features,
                benchmark_references + accepted_references,
                phash_distance=phash_distance,
                ssim_threshold=ssim_threshold,
            )
            if duplicate is not None:
                rows.append(
                    {
                        "source": source,
                        "original_stem": stem,
                        "stem": "",
                        "status": "removed_duplicate",
                        "duplicate_of": f"{duplicate['source']}:{duplicate['stem']}",
                        "duplicate_reason": reason,
                        "similarity": "" if score is None else score,
                        "sha256": features["sha256"],
                        "phash": f"{features['phash']:016x}",
                        "contrast": features["contrast"],
                        "lesion_ratio": features["lesion_ratio"],
                    }
                )
                continue

            materialized_stem = _safe_stem(source, stem)
            image_destination = images_out / f"{materialized_stem}{image_path.suffix.lower()}"
            mask_destination = masks_out / f"{materialized_stem}.png"
            shutil.copy2(image_path, image_destination)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {mask_path}")
            if not cv2.imwrite(str(mask_destination), np.where(mask > 127, 255, 0).astype(np.uint8)):
                raise OSError(f"Failed to write mask: {mask_destination}")
            reference = {
                **features,
                "source": source,
                "stem": materialized_stem,
                "path": str(image_destination),
                "is_benchmark": False,
            }
            accepted_references.append(reference)
            rows.append(
                {
                    "source": source,
                    "original_stem": stem,
                    "stem": materialized_stem,
                    "status": "accepted",
                    "duplicate_of": "",
                    "duplicate_reason": "",
                    "similarity": "",
                    "sha256": features["sha256"],
                    "phash": f"{features['phash']:016x}",
                    "contrast": features["contrast"],
                    "lesion_ratio": features["lesion_ratio"],
                }
            )

    if not rows:
        raise ValueError("No source samples were discovered.")
    _attach_strata(rows)
    manifest_path = output_root / "data_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"manifest": manifest_path, "accepted": sum(row["status"] == "accepted" for row in rows), "rows": rows}


def _tertile(values, value):
    ordered = sorted(float(item) for item in values)
    low = ordered[math.floor((len(ordered) - 1) / 3)]
    high = ordered[math.floor(2 * (len(ordered) - 1) / 3)]
    return 0 if value <= low else 1 if value <= high else 2


def _attach_strata(rows):
    accepted = [row for row in rows if row["status"] == "accepted"]
    contrasts = [float(row["contrast"]) for row in accepted]
    lesion_ratios = [float(row["lesion_ratio"]) for row in accepted]
    for row in rows:
        if row["status"] != "accepted":
            row["contrast_bin"] = ""
            row["lesion_bin"] = ""
            row["stratum"] = ""
            continue
        contrast_bin = _tertile(contrasts, float(row["contrast"]))
        lesion_bin = _tertile(lesion_ratios, float(row["lesion_ratio"]))
        row["contrast_bin"] = contrast_bin
        row["lesion_bin"] = lesion_bin
        row["stratum"] = f"{row['source']}|c{contrast_bin}|l{lesion_bin}"


def read_manifest(path, accepted_only=True):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("status") == "accepted"] if accepted_only else rows
