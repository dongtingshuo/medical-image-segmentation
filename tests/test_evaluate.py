from pathlib import Path

import pytest

from evaluate import VALID_EVALUATION_SPLITS, resolve_split_paths


def test_evaluate_supports_external_split_paths(tmp_path):
    config = {
        "data": {
            "external_images_dir": str(tmp_path / "external" / "images"),
            "external_masks_dir": str(tmp_path / "external" / "masks"),
        }
    }

    images_path, masks_path = resolve_split_paths(config, "external")

    assert "external" in VALID_EVALUATION_SPLITS
    assert images_path == Path(config["data"]["external_images_dir"])
    assert masks_path == Path(config["data"]["external_masks_dir"])


def test_evaluate_rejects_unknown_split():
    with pytest.raises(ValueError, match="Unsupported evaluation split"):
        resolve_split_paths({"data": {}}, "holdout")
