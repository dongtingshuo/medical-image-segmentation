import cv2
import numpy as np

from src.data_preparation import prepare_external_split, prepare_internal_splits, scan_pairs


def _write_pair(root, split, sample_id):
    images = root / split / "images"
    masks = root / split / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images / f"{sample_id}.jpg"), np.full((20, 30, 3), 127, dtype=np.uint8))
    cv2.imwrite(str(masks / f"{sample_id}_segmentation.png"), np.full((20, 30), 255, dtype=np.uint8))


def test_prepare_internal_splits_preserves_independent_test(tmp_path):
    source = tmp_path / "source"
    for split in ("train", "val", "test"):
        _write_pair(source, split, f"isic_{split}")

    pairs, report = scan_pairs(source)
    assert report["pair_counts"] == {"test": 1, "train": 1, "val": 1}
    assert pairs["test"][0][0] == "isic_test"

    output = tmp_path / "prepared"
    prepared = prepare_internal_splits(source, output, image_size=16)
    assert prepared["prepared_splits"] == {"train": 1, "val": 1, "test": 1}
    assert cv2.imread(str(output / "test" / "images" / "isic_test.jpg")).shape[:2] == (16, 16)


def test_external_split_excludes_internal_ids(tmp_path):
    source = tmp_path / "external"
    _write_pair(source, "test", "overlap")
    _write_pair(source, "test", "external_only")

    output = tmp_path / "prepared_external"
    report = prepare_external_split(source, output, excluded_ids={"overlap"}, image_size=16)
    assert report["excluded_overlap_count"] == 1
    assert report["prepared_pairs"] == 1
    assert (output / "images" / "external_only.jpg").exists()
    assert not (output / "images" / "overlap.jpg").exists()
