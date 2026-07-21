import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

import scripts.run_v1_6_pipeline as v16_pipeline
from scripts.run_v1_6_pipeline import (
    _materialize_ham_pretrain,
    _materialize_subset,
    _prune_oof_candidate_payload,
    _select_oof_candidate,
    package_state,
    prepare_evaluation_data,
    prune_runtime_data_for_oof,
    reset_teacher_training,
    restore_state_archive,
)
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
                    "stem": f"{source}__{source}_{index}",
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


def test_ham_pretrain_split_derives_materialized_ids_from_source_metadata():
    rows = _records()
    for row in rows:
        if row["source"] == "ham10000":
            row["stem"] = row["original_stem"]
    split = create_ham_pretrain_split(rows, validation_fraction=0.20)
    assert all(stem.startswith("ham10000__") for stem in split["train_ids"] + split["val_ids"])


def test_materialize_subset_resolves_unique_legacy_original_stem(tmp_path):
    images, masks, output = tmp_path / "images", tmp_path / "masks", tmp_path / "output"
    images.mkdir()
    masks.mkdir()
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cv2.imwrite(str(images / "ham10000__HAM_0000008.jpg"), image)
    cv2.imwrite(str(masks / "ham10000__HAM_0000008.png"), mask)
    _materialize_subset(images, masks, {"train": ["HAM_0000008"]}, output)
    assert (output / "train/images/ham10000__HAM_0000008.jpg").exists()
    assert (output / "train/masks/ham10000__HAM_0000008.png").exists()


def test_ham_pretraining_materialization_uses_original_source_stem(tmp_path):
    images, masks, output = tmp_path / "images", tmp_path / "masks", tmp_path / "output"
    images.mkdir()
    masks.mkdir()
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cv2.imwrite(str(images / "ISIC_0000008.jpg"), image)
    cv2.imwrite(str(masks / "ISIC_0000008.png"), mask)
    cv2.imwrite(str(images / "ISIC_0000009.jpg"), image)
    cv2.imwrite(str(masks / "ISIC_0000009.png"), mask)
    records = [
        {
            "source": "ham10000",
            "original_stem": "ISIC_0000008",
            "stem": "HAM_0000008",
            "status": "accepted",
            "group_id": "lesion_1",
        },
        {
            "source": "ham10000",
            "original_stem": "ISIC_0000009",
            "stem": "HAM_0000009",
            "status": "accepted",
            "group_id": "lesion_2",
        },
    ]
    counts = _materialize_ham_pretrain(images, masks, records, {"lesion_1"}, output)
    assert counts == {"train": 1, "val": 1}
    assert (output / "val/images/ham10000__ISIC_0000008.jpg").exists()
    assert (output / "train/masks/ham10000__ISIC_0000009.png").exists()


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
    assert not (root / "prepared").exists()
    assert not (root / "target_train_data").exists()


def test_v16_oof_cleanup_preserves_only_runtime_data_needed_by_students(monkeypatch, tmp_path):
    root = tmp_path / "research_v1_6"
    removed = (
        "prepared/raw.jpg",
        "fold_data/fold_0/train/images/raw.jpg",
        "pretrain_data/train/images/raw.jpg",
        "validation_probability_cache/raw.npy",
        "streaming/raw.npy",
    )
    kept = (
        "merged_train/images/raw.jpg",
        "target_train_data/train/images/raw.jpg",
        "pretrain_data/val/images/raw.jpg",
    )
    for relative in removed + kept:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    monkeypatch.setattr(
        v16_pipeline.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=5 * 1024**3),
    )

    prune_runtime_data_for_oof(root)

    assert all(not (root / relative).exists() for relative in removed)
    assert all((root / relative).exists() for relative in kept)


def test_v16_selection_prepares_only_locked_evaluation_splits(monkeypatch, tmp_path):
    output_root = tmp_path / "research_v1_6"
    calls = []

    def fake_internal(_source, destination, image_size):
        calls.append(("internal", image_size))
        for relative in ("val/images", "val/masks", "test/images", "test/masks"):
            (Path(destination) / relative).mkdir(parents=True, exist_ok=True)

    def fake_external(_source, destination, excluded_ids, image_size):
        calls.append(("external", excluded_ids, image_size))
        for relative in ("images", "masks"):
            (Path(destination) / relative).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(v16_pipeline, "prepare_internal_splits", fake_internal)
    monkeypatch.setattr(v16_pipeline, "prepare_external_split", fake_external)
    monkeypatch.setattr(v16_pipeline, "collect_sample_ids", lambda _root: {"held-out"})
    args = SimpleNamespace(isic17_root="isic17", isic18_root="isic18")

    prepared = prepare_evaluation_data(args, output_root)

    assert prepared == output_root / "prepared"
    assert calls == [("internal", None), ("external", {"held-out"}, None)]
    assert not (output_root / "merged_train").exists()
    assert not (output_root / "pretrain_data").exists()
    assert not (output_root / "target_train_data").exists()


def test_v16_oof_candidate_pruning_keeps_selection_metadata(tmp_path):
    candidate = tmp_path / "candidate"
    (candidate / "soft_masks").mkdir(parents=True)
    (candidate / "family_selection.json").write_text("{}", encoding="utf-8")
    (candidate / "unetpp_crossfit.npy").write_bytes(b"probabilities")
    (candidate / "segformer_crossfit.npy").write_bytes(b"probabilities")
    (candidate / "targets_384.npy").write_bytes(b"targets")
    (candidate / "soft_masks/sample.png").write_bytes(b"mask")

    _prune_oof_candidate_payload(candidate)

    assert (candidate / "family_selection.json").exists()
    assert not (candidate / "unetpp_crossfit.npy").exists()
    assert not (candidate / "segformer_crossfit.npy").exists()
    assert not (candidate / "targets_384.npy").exists()
    assert not (candidate / "soft_masks").exists()


def test_v16_oof_generation_keeps_only_current_selected_candidate(monkeypatch, tmp_path):
    root = tmp_path / "research_v1_6"
    metrics = {
        "none": {"dice": 0.8000, "boundary_f1": 0.5000},
        "flip": {"dice": 0.8020, "boundary_f1": 0.5000},
        "multiscale_flip": {"dice": 0.8005, "boundary_f1": 0.6000},
    }

    def fake_run(command):
        candidate = Path(command[command.index("--output-root") + 1])
        mode = command[command.index("--tta") + 1]
        candidate.mkdir(parents=True, exist_ok=True)
        for architecture in ("unetpp", "segformer"):
            np.save(candidate / f"{architecture}_crossfit.npy", np.zeros((1, 1, 2, 2), dtype=np.float16))
        np.save(candidate / "targets_384.npy", np.zeros((1, 1, 2, 2), dtype=np.uint8))
        (candidate / "family_selection.json").write_text(
            json.dumps({"oof_metrics": metrics[mode]}),
            encoding="utf-8",
        )

    def fake_select(candidates, _output_root):
        assert not (candidates["none"] / "unetpp_crossfit.npy").exists()
        assert (candidates["flip"] / "unetpp_crossfit.npy").exists()
        assert not (candidates["multiscale_flip"] / "unetpp_crossfit.npy").exists()

    monkeypatch.setattr(v16_pipeline, "prune_runtime_data_for_oof", lambda _root: None)
    monkeypatch.setattr(v16_pipeline, "run", fake_run)
    monkeypatch.setattr(v16_pipeline, "_select_oof_candidate", fake_select)
    state = {"completed": []}

    assert v16_pipeline.run_oof(root, state)
    assert state["completed"] == ["oof_generated"]
    assert not (root / "oof/candidates").exists()


def test_v16_state_package_prunes_only_non_resumable_completed_checkpoints(tmp_path):
    root = tmp_path / "research_v1_6"
    state = {
        "version": "1.6",
        "phase": "teachers_unetpp",
        "completed": ["model:teacher-unetpp-fold0:adapt", "model:student-unetpp:student"],
    }
    root.mkdir()
    (root / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    for task in ("teacher-unetpp-fold0", "teacher-unetpp-fold1", "student-unetpp"):
        checkpoint_dir = root / "models" / task / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best_model.pth").write_bytes(f"best:{task}".encode())
        (checkpoint_dir / "last_model.pth").write_bytes(f"last:{task}".encode())

    archive = package_state(root)
    restored = tmp_path / "restored"
    restore_state_archive(archive, restored)

    assert (restored / "models/teacher-unetpp-fold0/checkpoints/best_model.pth").exists()
    assert not (restored / "models/teacher-unetpp-fold0/checkpoints/last_model.pth").exists()
    assert (restored / "models/teacher-unetpp-fold1/checkpoints/best_model.pth").exists()
    assert (restored / "models/teacher-unetpp-fold1/checkpoints/last_model.pth").exists()
    assert not (restored / "models/student-unetpp/checkpoints/best_model.pth").exists()
    assert (restored / "models/student-unetpp/checkpoints/last_model.pth").exists()


def test_v16_kaggle_installs_do_not_retain_pip_cache():
    notebook = open("notebooks/kaggle_v1_6.py", encoding="utf-8").read()
    gpu_setup = open("scripts/kaggle_prepare_gpu.py", encoding="utf-8").read()
    assert '"--no-cache-dir"' in notebook
    assert '"--no-cache-dir"' in gpu_setup


def test_v16_teacher_reset_preserves_pretraining_and_removes_downstream_state(tmp_path):
    root = tmp_path / "research_v1_6"
    for name in ("pretrain-unetpp", "teacher-unetpp-fold0", "teacher-segformer-fold0", "student-unetpp"):
        path = root / "models" / name
        path.mkdir(parents=True)
        (path / "artifact.bin").write_bytes(b"state")
    for name in ("oof", "selection", "final", "release"):
        (root / name).mkdir(parents=True)
    state = {
        "phase": "students",
        "completed": [
            "data_prepared",
            "model:pretrain-unetpp:pretrain",
            "model:teacher-unetpp-fold0:mixed",
            "model:teacher-unetpp-fold0:adapt",
            "model:student-unetpp:student",
            "oof_generated",
        ],
        "failed": {"teacher-segformer-fold0:mixed": "network", "unrelated": "keep"},
    }

    reset_teacher_training(root, state)

    assert (root / "models/pretrain-unetpp").exists()
    assert not (root / "models/teacher-unetpp-fold0").exists()
    assert not (root / "models/teacher-segformer-fold0").exists()
    assert not (root / "models/student-unetpp").exists()
    assert not (root / "oof").exists()
    assert state["phase"] == "teachers_unetpp"
    assert state["completed"] == ["data_prepared", "model:pretrain-unetpp:pretrain"]
    assert state["failed"] == {"unrelated": "keep"}
    assert state["teacher_restart_reason"] == "reset_seed_epoch_and_optimizer_state"


def test_v16_teacher_seed_resets_epoch_and_optimizer_state(monkeypatch, tmp_path):
    seed_dir = tmp_path / "pretrain/checkpoints"
    seed_dir.mkdir(parents=True)
    seed_last = seed_dir / "last_model.pth"
    seed_best = seed_dir / "best_model.pth"
    seed_last.write_bytes(b"last")
    seed_best.write_bytes(b"best")
    captured = {}

    def fake_run(command):
        captured["command"] = [str(part) for part in command]
        runtime_config = yaml.safe_load(
            (tmp_path / "models/teacher-unetpp-fold0/runtime_config.yaml").read_text(encoding="utf-8")
        )
        captured["training"] = runtime_config["training"]

    class Budget:
        @staticmethod
        def can_start():
            return True

        @staticmethod
        def remaining_minutes():
            return 100.0

    monkeypatch.setattr(v16_pipeline, "run", fake_run)
    monkeypatch.setattr(v16_pipeline, "_finished", lambda _path, _epochs: True)
    base = {
        "data": {},
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "resume_optimizer": True,
            "resume_training_state": True,
            "early_stopping": {"enabled": True},
        },
    }
    state = {"completed": [], "failed": {}}

    completed = v16_pipeline.train_stage(
        "teacher-unetpp-fold0",
        "unetpp",
        base,
        tmp_path,
        (tmp_path / "train/images", tmp_path / "train/masks"),
        (tmp_path / "val/images", tmp_path / "val/masks"),
        tmp_path / "manifest.csv",
        Budget(),
        state,
        18,
        "mixed",
        seed_checkpoint=(seed_last, seed_best),
    )

    assert completed
    assert captured["training"]["resume_optimizer"] is False
    assert captured["training"]["resume_training_state"] is False
    assert "--resume" in captured["command"]
    assert (tmp_path / "models/teacher-unetpp-fold0/checkpoints/last_model.pth").read_bytes() == b"best"


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
