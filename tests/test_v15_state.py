import json
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.run_v1_5_pipeline import package_release, package_state, prepare_data, restore_state_archive


def _write_source_pair(root, split, stem, seed):
    images = root / split / "images"
    masks = root / split / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4 + seed % 3 : 18, 6:25] = 255
    cv2.imwrite(str(images / f"{stem}.jpg"), image)
    cv2.imwrite(str(masks / f"{stem}_segmentation.png"), mask)
    return images, masks


def test_state_package_has_sha256_and_excludes_recomputable_fold_data(tmp_path):
    root = tmp_path / "research_v1_5"
    (root / "models").mkdir(parents=True)
    (root / "fold_data/fold_0").mkdir(parents=True)
    (root / "merged_train/images").mkdir(parents=True)
    (root / "prepared/internal/val/images").mkdir(parents=True)
    (root / "oof/soft_masks").mkdir(parents=True)
    (root / "pipeline_state.json").write_text(json.dumps({"phase": "teachers"}), encoding="utf-8")
    (root / "models/model.pth").write_bytes(b"weights")
    (root / "fold_data/fold_0/recomputed.jpg").write_bytes(b"data")
    (root / "merged_train/images/raw.jpg").write_bytes(b"raw")
    (root / "prepared/internal/val/images/raw.jpg").write_bytes(b"raw")
    (root / "oof/soft_masks/teacher-output.png").write_bytes(b"prediction")
    archive = package_state(root)
    assert archive.exists()
    assert archive.with_suffix(".zip.sha256").exists()
    restored = tmp_path / "restored"
    restore_state_archive(archive, restored)
    assert (restored / "models/model.pth").exists()
    assert not (restored / "fold_data").exists()
    assert not (restored / "merged_train/images/raw.jpg").exists()
    assert not (restored / "prepared/internal/val/images/raw.jpg").exists()
    assert (restored / "oof/soft_masks/teacher-output.png").exists()


def test_state_restore_rejects_corrupt_archive(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "pipeline_state.json").write_text("{}", encoding="utf-8")
    archive = package_state(root)
    archive.with_suffix(".zip.sha256").write_text("0" * 64 + "  v1_5_state.zip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        restore_state_archive(archive, tmp_path / "restored")


def test_release_package_contains_only_locked_publishable_variants(tmp_path):
    root = tmp_path / "research"
    model = root / "models/student-unetpp"
    model.mkdir(parents=True)
    (model / "best_model.pth").write_bytes(b"weights")
    (model / "runtime_config.yaml").write_text(
        "model:\n  model_name: unet_plus_plus\ndata:\n  image_size: 384\n",
        encoding="utf-8",
    )
    members = root / "members.json"
    members.write_text(
        json.dumps(
            [
                {
                    "name": "student-unetpp",
                    "config": str(model / "runtime_config.yaml"),
                    "checkpoint": str(model / "best_model.pth"),
                }
            ]
        ),
        encoding="utf-8",
    )
    selection = root / "selection"
    selection.mkdir()
    decision = selection / "locked_decision.json"
    decision.write_text(
        json.dumps(
            {
                "fast": {"member": "student-unetpp", "threshold": 0.4},
                "members": ["student-unetpp"],
                "threshold": 0.4,
            }
        ),
        encoding="utf-8",
    )
    (root / "final").mkdir()
    (root / "final/evaluation_complete.json").write_text(
        json.dumps({"publish_default": True, "publish_best_accuracy": True}), encoding="utf-8"
    )
    (root / "merged_train").mkdir()
    (root / "merged_train/data_manifest.csv").write_text("stem,status\na,accepted\n", encoding="utf-8")
    (root / "data_sources.json").write_text("{}", encoding="utf-8")
    archive = package_release(root, members, decision)
    assert archive.exists()
    assert (root / "release/fast/best_model.pth").exists()
    assert (root / "release/best_accuracy/members/student-unetpp/best_model.pth").exists()
    manifest = json.loads((root / "release/release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["published_variants"] == ["fast", "best_accuracy"]


def test_prepared_data_rematerializes_to_identical_manifest(tmp_path):
    isic17 = tmp_path / "isic17"
    for index in range(6):
        split = "train" if index < 5 else "val"
        _write_source_pair(isic17, split, f"isic17_{index}", index)
    _write_source_pair(isic17, "test", "isic17_test", 20)
    isic18 = tmp_path / "isic18"
    _write_source_pair(isic18, "test", "isic18_test", 30)
    isic16 = _write_source_pair(tmp_path / "isic16", "all", "isic16_0", 40)
    ph2 = _write_source_pair(tmp_path / "ph2", "all", "ph2_0", 50)
    args = SimpleNamespace(
        isic17_root=isic17,
        isic18_root=isic18,
        isic16_images=isic16[0],
        isic16_masks=isic16[1],
        ph2_images=ph2[0],
        ph2_masks=ph2[1],
    )
    output = tmp_path / "output"
    state = {"completed": [], "phase": "data"}
    prepare_data(args, output, state)
    first_hash = state["data_manifest_sha256"]
    state["phase"] = "teachers"
    shutil.rmtree(output / "merged_train/images")
    shutil.rmtree(output / "prepared/internal/val/images")
    prepare_data(args, output, state)
    assert state["phase"] == "teachers"
    assert state["data_manifest_sha256"] == first_hash
    assert (output / "fold_data/fold_0/train/images").exists()
