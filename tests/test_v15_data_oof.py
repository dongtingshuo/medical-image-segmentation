import csv

import cv2
import numpy as np

from src.cross_validation import create_stratified_kfold_splits, validate_folds
from src.multisource_data import discover_pairs, prepare_multisource_dataset
from src.oof import difficulty_weight, write_oof_outputs


def _write_pair(root, stem, value, shape=(20, 30)):
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    image = np.full((*shape, 3), value, dtype=np.uint8)
    image[:, shape[1] // 2 :] = min(value + 10, 255)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:-4, 5:-5] = 255
    cv2.imwrite(str(images / f"{stem}.jpg"), image)
    cv2.imwrite(str(masks / f"{stem}.png"), mask)
    return images, masks


def test_multisource_dedup_removes_added_benchmark_overlap(tmp_path):
    primary = _write_pair(tmp_path / "primary", "p", 40)
    extra = _write_pair(tmp_path / "extra", "e", 120)
    benchmark = _write_pair(tmp_path / "benchmark", "b", 120)
    result = prepare_multisource_dataset(
        [("isic17", *primary), ("isic16", *extra)],
        [("test", *benchmark)],
        tmp_path / "merged",
    )
    removed = [row for row in result["rows"] if row["status"] == "removed_duplicate"]
    assert len(removed) == 1
    assert removed[0]["source"] == "isic16"
    assert cv2.imread(str(tmp_path / "merged/masks/isic17__p.png"), cv2.IMREAD_GRAYSCALE) is not None


def test_recursive_shared_root_discovers_nested_images_and_lesion_masks(tmp_path):
    root = tmp_path / "dataset"
    images, masks = _write_pair(root / "Training", "sample_train", 50)
    nested_image = root / "Training_Data/sample_train.jpg"
    nested_mask = root / "Training_GroundTruth/sample_train_lesion.png"
    nested_image.parent.mkdir(parents=True)
    nested_mask.parent.mkdir(parents=True)
    (images / "sample_train.jpg").replace(nested_image)
    (masks / "sample_train.png").replace(nested_mask)
    pairs = discover_pairs(root, root)
    assert [(stem, image.name, mask.name) for stem, image, mask in pairs] == [
        ("sample_train", "sample_train.jpg", "sample_train_lesion.png")
    ]


def test_recursive_shared_root_discovers_ph2_bmp_pairs(tmp_path):
    root = tmp_path / "PH2Dataset"
    image_dir = root / "PH2 Dataset images/IMD002/IMD002_Dermoscopic_Image"
    mask_dir = root / "PH2 Dataset images/IMD002/IMD002_lesion"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.full((20, 30, 3), 80, dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:-4, 5:-5] = 255
    cv2.imwrite(str(image_dir / "IMD002.bmp"), image)
    cv2.imwrite(str(mask_dir / "IMD002_lesion.bmp"), mask)

    pairs = discover_pairs(root, root)

    assert [(stem, image.name, mask.name) for stem, image, mask in pairs] == [
        ("IMD002", "IMD002.bmp", "IMD002_lesion.bmp")
    ]


def test_multisource_always_removes_primary_benchmark_overlap(tmp_path):
    primary = _write_pair(tmp_path / "primary", "p", 80)
    benchmark = _write_pair(tmp_path / "benchmark", "b", 80)
    result = prepare_multisource_dataset([("isic17", *primary)], [("test", *benchmark)], tmp_path / "merged")
    assert result["accepted"] == 0
    assert result["rows"][0]["status"] == "removed_duplicate"


def test_stratified_folds_partition_every_sample_once():
    records = [
        {"stem": f"s{index}", "stratum": f"source{index % 3}|c{index % 2}|l{index % 3}"}
        for index in range(30)
    ]
    folds = create_stratified_kfold_splits(records, k=5, seed=7)
    assert validate_folds(folds, [record["stem"] for record in records])
    assert all(fold["val_ids"] for fold in folds)


def test_oof_outputs_require_both_architectures_and_preserve_geometry(tmp_path):
    images, masks = _write_pair(tmp_path / "source", "a", 60, shape=(20, 30))
    _write_pair(tmp_path / "source", "b", 90, shape=(18, 24))
    folds = [
        {"fold": 0, "train_ids": ["b"], "val_ids": ["a"]},
        {"fold": 1, "train_ids": ["a"], "val_ids": ["b"]},
    ]
    prediction = np.zeros((16, 16), dtype=np.float32)
    prediction[4:12, 4:12] = 0.9
    predictions = {
        "unetpp": {"a": prediction, "b": prediction},
        "manet": {"a": prediction * 0.95, "b": prediction * 0.95},
    }
    manifest = [
        {"stem": "a", "contrast_bin": "0"},
        {"stem": "b", "contrast_bin": "2"},
    ]
    result = write_oof_outputs(predictions, folds, manifest, images, masks, tmp_path / "oof")
    soft_a = cv2.imread(str(result["soft_masks_dir"] / "a.png"), cv2.IMREAD_UNCHANGED)
    assert soft_a.shape == (20, 30)
    assert soft_a.dtype == np.uint16
    with result["weights"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(1.0 <= float(row["weight"]) <= 3.0 for row in rows)
    assert difficulty_weight(0.0, 0.0, 1.0)[0] == 3.0
