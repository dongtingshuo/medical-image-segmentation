import csv
import json

import cv2
import numpy as np
import pytest
import torch

from scripts.run_v1_6_pipeline import _select_oof_candidate, package_state, restore_state_archive
from src.dataset import SkinLesionDataset
from src.losses import build_loss
from src.v16 import (
    create_ham_pretrain_split,
    create_target_domain_folds,
    decorate_v16_manifest,
    select_crossfit_family_weights,
    validate_target_domain_folds,
)


def _records():
    rows = []
    for source, count in (("isic17", 10), ("isic16", 2), ("ph2", 2), ("ham10000", 6)):
        for index in range(count):
            rows.append(
                {
                    "source": source,
                    "original_stem": f"{source}_{index}",
                    "stem": f"{source}__{index}",
                    "status": "accepted",
                    "stratum": f"{source}|{index % 3}|{index % 3}",
                    "group_id": f"lesion_{index // 2}" if source == "ham10000" else f"{source}_{index}",
                }
            )
    return rows


def test_target_domain_folds_only_validate_isic17_and_exclude_ham_from_train():
    rows = _records()
    folds = create_target_domain_folds(rows, k=5, seed=42)
    assert validate_target_domain_folds(folds, rows)
    assert all(stem.startswith("isic17__") for fold in folds for stem in fold["val_ids"])
    assert all(not stem.startswith("ham10000__") for fold in folds for stem in fold["train_ids"])


def test_ham_pretrain_split_keeps_lesion_groups_together():
    rows = _records()
    split = create_ham_pretrain_split(rows, validation_fraction=0.20)
    group_by_stem = {row["stem"]: row["group_id"] for row in rows}
    train_groups = {group_by_stem[stem] for stem in split["train_ids"]}
    val_groups = {group_by_stem[stem] for stem in split["val_ids"]}
    assert split["train_ids"] and split["val_ids"]
    assert not set(split["train_ids"]) & set(split["val_ids"])
    assert not train_groups & val_groups


def test_manifest_decoration_records_ham_license_and_group(tmp_path):
    metadata = tmp_path / "HAM10000_metadata.csv"
    metadata.write_text("image_id,lesion_id\nham_a,lesion_a\n", encoding="utf-8")
    manifest = tmp_path / "data_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "original_stem", "stem", "status"])
        writer.writeheader()
        writer.writerow({"source": "ham10000", "original_stem": "ham_a", "stem": "ham10000__ham_a", "status": "accepted"})
        writer.writerow({"source": "isic17", "original_stem": "isic_a", "stem": "isic17__isic_a", "status": "accepted"})
    rows = decorate_v16_manifest(manifest, metadata)
    assert rows[0]["label_quality"] == "reviewed"
    assert rows[0]["stem"] == "ham10000__ham_a"
    assert rows[0]["group_id"] == "lesion_a"
    assert rows[0]["license"] == "CC BY-NC-SA 4.0"
    assert rows[1]["stage"] == "finetune"


def test_v16_losses_backpropagate_and_gate_low_confidence_pixels():
    logits = torch.randn(2, 1, 12, 12, requires_grad=True)
    targets = torch.randint(0, 2, logits.shape).float()
    soft = torch.full_like(targets, 0.5)
    pretrain = build_loss({"loss": {"name": "bootstrapped_bce_dice", "beta": 0.85}})
    pretrain(logits, targets).backward(retain_graph=True)
    assert torch.isfinite(logits.grad).all()
    logits.grad.zero_()
    student = build_loss({"loss": {"name": "confidence_gated_distillation", "confidence_threshold": 0.60}})
    student(logits, targets, soft).backward()
    assert torch.isfinite(logits.grad).all()


def test_missing_soft_masks_can_be_neutral_for_confidence_gated_distillation(tmp_path):
    images, masks, soft = tmp_path / "images", tmp_path / "masks", tmp_path / "soft"
    images.mkdir()
    masks.mkdir()
    soft.mkdir()
    image = np.full((12, 14, 3), 127, dtype=np.uint8)
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[3:9, 4:10] = 255
    cv2.imwrite(str(images / "sample.jpg"), image)
    cv2.imwrite(str(masks / "sample.png"), mask)
    dataset = SkinLesionDataset(images, masks, soft_masks_dir=soft, missing_soft_mask_value=0.5)
    _, _, neutral = dataset[0]
    assert np.allclose(neutral.numpy(), 0.5)


def test_crossfit_family_weight_selection_prefers_higher_dice_family():
    targets = np.zeros((2, 1, 8, 8), dtype=np.float32)
    targets[:, :, 2:6, 2:6] = 1.0
    unet = targets * 0.9 + (1.0 - targets) * 0.1
    segformer = np.full_like(unet, 0.45)
    selected, rows = select_crossfit_family_weights({"unetpp": unet, "segformer": segformer}, targets)
    assert selected["unetpp_weight"] == 1.0
    assert len(rows) == 21


def test_v16_state_archive_has_versioned_name_and_checksum(tmp_path):
    root = tmp_path / "research_v1_6"
    root.mkdir()
    (root / "pipeline_state.json").write_text(json.dumps({"version": "1.6", "phase": "pretrain"}), encoding="utf-8")
    (root / "models").mkdir()
    (root / "models/model.pth").write_bytes(b"weights")
    (root / "prepared/raw.jpg").parent.mkdir(parents=True)
    (root / "prepared/raw.jpg").write_bytes(b"raw")
    (root / "target_train_data/train/images/raw.jpg").parent.mkdir(parents=True)
    (root / "target_train_data/train/images/raw.jpg").write_bytes(b"raw")
    archive = package_state(root)
    assert archive.name == "v1_6_state.zip"
    assert archive.with_suffix(".zip.sha256").exists()
    restored = tmp_path / "restored"
    restore_state_archive(archive, restored)
    assert (restored / "models/model.pth").exists()
    assert not (restored / "prepared/raw.jpg").exists()
    assert not (restored / "target_train_data/train/images/raw.jpg").exists()


def test_manifest_rejects_accepted_ham_sample_without_metadata(tmp_path):
    metadata = tmp_path / "HAM10000_metadata.csv"
    metadata.write_text("image_id,lesion_id\nother,lesion\n", encoding="utf-8")
    manifest = tmp_path / "data_manifest.csv"
    manifest.write_text("source,original_stem,stem,status\nham10000,missing,ham10000__missing,accepted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from metadata"):
        decorate_v16_manifest(manifest, metadata)


def test_oof_tta_selection_is_locked_before_validation(tmp_path):
    root = tmp_path / "research"
    image_dir = root / "merged_train/images"
    image_dir.mkdir(parents=True)
    for stem in ("isic17__a", "isic17__b"):
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), np.full((8, 8, 3), 120, dtype=np.uint8))
    targets = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    targets[:, :, 2:6, 2:6] = 1
    candidates = {}
    for mode in ("none", "flip", "multiscale_flip"):
        candidate = root / "oof/candidates" / mode
        (candidate / "soft_masks").mkdir(parents=True)
        for architecture in ("unetpp", "segformer"):
            probability = targets.astype(np.float16) * 0.9 + (1 - targets.astype(np.float16)) * 0.1
            np.save(candidate / f"{architecture}_crossfit.npy", probability)
        np.save(candidate / "targets_384.npy", targets)
        (candidate / "family_selection.json").write_text(
            json.dumps(
                {
                    "family_weights": {"unetpp": 0.5, "segformer": 0.5},
                    "oof_metrics": {"dice": 0.90 if mode == "none" else 0.9005, "boundary_f1": 0.80},
                    "stems": ["isic17__a", "isic17__b"],
                    "resize_mode": "stretch",
                }
            ),
            encoding="utf-8",
        )
        (candidate / "soft_masks/sample.png").write_bytes(b"mask")
        candidates[mode] = candidate
    _select_oof_candidate(candidates, root)
    decision = json.loads((root / "oof/family_selection.json").read_text(encoding="utf-8"))
    assert decision["tta"] == "none"
    assert (root / "oof/soft_masks/isic17__a.png").exists()
