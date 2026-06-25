from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def paired_stems(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory does not exist: {masks_dir}")
    images = {path.stem: path for path in sorted(images_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {path.stem: path for path in sorted(masks_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS}
    missing_masks = sorted(set(images) - set(masks))
    extra_masks = sorted(set(masks) - set(images))
    if missing_masks or extra_masks:
        raise ValueError(
            "Image/mask stems do not match. "
            f"missing_masks={missing_masks[:10]}, extra_masks={extra_masks[:10]}"
        )
    stems = sorted(images)
    if not stems:
        raise ValueError(f"No paired image/mask files found in {images_dir} and {masks_dir}")
    return stems


def create_kfold_splits(stems, k=3, seed=42):
    stems = sorted(str(stem) for stem in stems)
    k = int(k)
    if k < 2:
        raise ValueError("k must be at least 2.")
    if len(stems) < k:
        raise ValueError(f"Need at least k samples to create folds. samples={len(stems)}, k={k}")
    rng = random.Random(int(seed))
    shuffled = list(stems)
    rng.shuffle(shuffled)
    fold_sizes = [len(shuffled) // k] * k
    for index in range(len(shuffled) % k):
        fold_sizes[index] += 1
    folds = []
    cursor = 0
    all_stems = set(stems)
    for fold_index, fold_size in enumerate(fold_sizes):
        val_ids = sorted(shuffled[cursor : cursor + fold_size])
        train_ids = sorted(all_stems - set(val_ids))
        cursor += fold_size
        folds.append({"fold": fold_index, "train_ids": train_ids, "val_ids": val_ids})
    validate_folds(folds, stems)
    return folds


def validate_folds(folds, stems):
    expected = set(stems)
    validation_seen = []
    for fold in folds:
        train_ids = set(fold["train_ids"])
        val_ids = set(fold["val_ids"])
        if train_ids & val_ids:
            raise ValueError(f"Fold {fold.get('fold')} has train/val leakage: {sorted(train_ids & val_ids)[:10]}")
        if train_ids | val_ids != expected:
            raise ValueError(f"Fold {fold.get('fold')} does not cover all samples.")
        validation_seen.extend(val_ids)
    if sorted(validation_seen) != sorted(expected):
        duplicates = sorted({item for item in validation_seen if validation_seen.count(item) > 1})
        missing = sorted(expected - set(validation_seen))
        raise ValueError(f"Validation folds are not a partition. duplicates={duplicates[:10]}, missing={missing[:10]}")
    return True


def write_folds(path, folds, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "folds": folds}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_folds(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["folds"]


def materialize_fold_directories(images_dir, masks_dir, folds, output_root):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_root = Path(output_root)
    stem_to_image = {path.stem: path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    stem_to_mask = {path.stem: path for path in masks_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    for fold in folds:
        fold_root = output_root / f"fold_{int(fold['fold'])}"
        for split, ids in (("train", fold["train_ids"]), ("val", fold["val_ids"])):
            image_out = fold_root / split / "images"
            mask_out = fold_root / split / "masks"
            image_out.mkdir(parents=True, exist_ok=True)
            mask_out.mkdir(parents=True, exist_ok=True)
            for stem in ids:
                image_src = stem_to_image[stem]
                mask_src = stem_to_mask[stem]
                shutil.copy2(image_src, image_out / image_src.name)
                shutil.copy2(mask_src, mask_out / mask_src.name)
    return output_root
