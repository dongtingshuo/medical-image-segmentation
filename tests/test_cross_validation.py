import json

import pytest

from src.cross_validation import (
    create_kfold_splits,
    materialize_fold_directories,
    paired_stems,
    read_folds,
    validate_folds,
    write_folds,
)


def _touch_pair(root, stem):
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    (images / f"{stem}.jpg").write_text("image", encoding="utf-8")
    (masks / f"{stem}.png").write_text("mask", encoding="utf-8")
    return images, masks


def test_create_kfold_splits_has_no_leakage_and_full_val_partition():
    stems = [f"sample_{index:02d}" for index in range(9)]
    folds = create_kfold_splits(stems, k=3, seed=7)

    validate_folds(folds, stems)
    assert len(folds) == 3
    validation_stems = sorted(stem for fold in folds for stem in fold["val_ids"])
    assert validation_stems == stems
    for fold in folds:
        assert set(fold["train_ids"]).isdisjoint(fold["val_ids"])


def test_folds_round_trip_and_materialize_pairs(tmp_path):
    for stem in ["a", "b", "c", "d"]:
        images, masks = _touch_pair(tmp_path / "source", stem)
    stems = paired_stems(images, masks)
    folds = create_kfold_splits(stems, k=2, seed=1)

    fold_path = write_folds(tmp_path / "folds.json", folds, metadata={"k": 2})
    payload = json.loads(fold_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["k"] == 2
    assert read_folds(fold_path) == folds

    materialize_fold_directories(images, masks, folds, tmp_path / "fold_data")
    assert (tmp_path / "fold_data/fold_0/train/images").exists()
    assert (tmp_path / "fold_data/fold_0/val/masks").exists()


def test_materialize_fold_directories_supports_bmp_pairs(tmp_path):
    images = tmp_path / "source/images"
    masks = tmp_path / "source/masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    (images / "ph2__IMD002.bmp").write_bytes(b"image")
    (masks / "ph2__IMD002.bmp").write_bytes(b"mask")
    folds = [
        {"fold": 0, "train_ids": ["ph2__IMD002"], "val_ids": []},
        {"fold": 1, "train_ids": [], "val_ids": ["ph2__IMD002"]},
    ]

    output = materialize_fold_directories(images, masks, folds, tmp_path / "fold_data")

    assert (output / "fold_0/train/images/ph2__IMD002.bmp").read_bytes() == b"image"
    assert (output / "fold_1/val/masks/ph2__IMD002.bmp").read_bytes() == b"mask"


def test_materialize_fold_directories_reports_missing_manifest_stems(tmp_path):
    images, masks = _touch_pair(tmp_path / "source", "present")
    folds = [{"fold": 0, "train_ids": ["missing"], "val_ids": ["present"]}]

    with pytest.raises(ValueError, match="missing_images=\\['missing'\\]"):
        materialize_fold_directories(images, masks, folds, tmp_path / "fold_data")


def test_paired_stems_reports_mismatch(tmp_path):
    images, masks = _touch_pair(tmp_path / "source", "a")
    (images / "orphan.jpg").write_text("image", encoding="utf-8")

    with pytest.raises(ValueError, match="Image/mask stems do not match"):
        paired_stems(images, masks)
