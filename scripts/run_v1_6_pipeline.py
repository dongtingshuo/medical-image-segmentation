"""Five-session, target-domain v1.6 Kaggle segmentation workflow."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cross_validation import materialize_fold_directories, read_folds  # noqa: E402
from src.data_preparation import collect_sample_ids, prepare_external_split, prepare_internal_splits  # noqa: E402
from src.ensemble_v15 import macro_metrics, postprocess_masks  # noqa: E402
from src.multisource_data import discover_pairs, prepare_multisource_dataset  # noqa: E402
from src.oof import restore_probability, write_soft_mask  # noqa: E402
from src.utils import load_checkpoint_payload, load_config  # noqa: E402
from src.v16 import (  # noqa: E402
    HAM_LICENSE,
    create_ham_pretrain_split,
    create_target_domain_folds,
    decorate_v16_manifest,
    materialized_stem,
    write_v16_folds,
)

ARCHITECTURES = {
    "unetpp": {"model_name": "unet_plus_plus", "encoder_name": "efficientnet-b3", "batch_size": 6, "accumulation": 2},
    "segformer": {"model_name": "segformer", "encoder_name": "nvidia/mit-b3", "batch_size": 2, "accumulation": 6},
}
MIXED_PROPORTIONS = {"isic17": 0.70, "isic16": 0.20, "ph2": 0.10}
ADAPT_PROPORTIONS = {"isic17": 0.85, "isic16": 0.10, "ph2": 0.05}
RECOMPUTABLE_STATE_PATHS = (
    "prepared",
    "merged_train/images",
    "merged_train/masks",
    "fold_data",
    "pretrain_data",
    "target_train_data",
    "validation_probability_cache",
    "streaming",
    "oof/candidates",
)
OOF_RUNTIME_CLEANUP_PATHS = (
    "prepared",
    "fold_data",
    "pretrain_data/train",
    "validation_probability_cache",
    "streaming",
)
OOF_MIN_FREE_BYTES = 4 * 1024 * 1024 * 1024
STATE_PACKAGING_HEADROOM_BYTES = 256 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run(command):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def load_state(path):
    path = Path(path)
    if not path.exists():
        return {"version": "1.6", "session": 0, "phase": "data", "completed": [], "failed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def restore_state_archive(archive, output_root):
    if not archive:
        return
    archive = Path(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    if not archive.exists():
        raise FileNotFoundError(f"State archive does not exist: {archive}")
    if checksum.exists() and checksum.read_text(encoding="utf-8").split()[0] != sha256_file(archive):
        raise ValueError("State archive SHA256 mismatch.")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(output_root)


def _remove_recomputable_paths(output_root, relative_paths):
    output_root = Path(output_root)
    free_before = shutil.disk_usage(output_root.parent).free
    for relative in relative_paths:
        path = output_root / relative
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    free_after = shutil.disk_usage(output_root.parent).free
    released = max(0, free_after - free_before)
    if released:
        print(f"Released {released / (1024 ** 3):.2f} GiB of recomputable runtime data.", flush=True)


def prune_runtime_data_for_phase(output_root, phase):
    if phase not in {"teachers_unetpp", "teachers_segformer"}:
        return
    # Teacher training only needs the target-domain fold hardlinks and the
    # manifest. HAM pretraining and benchmark evaluation data are rebuilt in
    # later sessions, so retaining them here only consumes checkpoint space.
    _remove_recomputable_paths(
        output_root,
        (
            "prepared",
            "merged_train/images",
            "merged_train/masks",
            "pretrain_data",
            "target_train_data",
        ),
    )


def prune_runtime_data_for_oof(output_root):
    output_root = Path(output_root)
    _remove_recomputable_paths(output_root, OOF_RUNTIME_CLEANUP_PATHS)
    free_bytes = shutil.disk_usage(output_root.parent).free
    print(f"OOF free disk after cleanup: {free_bytes / (1024 ** 3):.2f} GiB.", flush=True)
    if free_bytes < OOF_MIN_FREE_BYTES:
        raise OSError(
            "Insufficient disk for streaming OOF generation after cleanup: "
            f"required={OOF_MIN_FREE_BYTES}, free={free_bytes}"
        )


def _prune_oof_candidate_payload(candidate_root):
    candidate_root = Path(candidate_root)
    for path in candidate_root.glob("*_crossfit.npy"):
        path.unlink(missing_ok=True)
    (candidate_root / "targets_384.npy").unlink(missing_ok=True)
    shutil.rmtree(candidate_root / "soft_masks", ignore_errors=True)


def _should_prune_checkpoint(relative, completed):
    if len(relative.parts) != 4 or relative.parts[0] != "models" or relative.parts[2] != "checkpoints":
        return False
    task_name = relative.parts[1]
    if task_name.startswith("teacher-"):
        return relative.name == "last_model.pth" and f"model:{task_name}:adapt" in completed
    if task_name.startswith("student-"):
        return relative.name == "best_model.pth" and f"model:{task_name}:student" in completed
    return False


def prune_redundant_checkpoints(output_root):
    output_root = Path(output_root)
    state_path = output_root / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    completed = set(state.get("completed", []))
    for path in sorted((output_root / "models").glob("*/checkpoints/*")):
        if path.is_file() and _should_prune_checkpoint(path.relative_to(output_root), completed):
            path.unlink(missing_ok=True)


def prune_completed_pipeline_state(output_root):
    output_root = Path(output_root)
    state_path = output_root / "pipeline_state.json"
    state = load_state(state_path)
    if state.get("phase") != "complete":
        return
    shutil.rmtree(output_root / "models", ignore_errors=True)
    shutil.rmtree(output_root / "oof/soft_masks", ignore_errors=True)
    shutil.rmtree(output_root / "selection/validation_probability_cache", ignore_errors=True)
    shutil.rmtree(output_root / "final/streaming", ignore_errors=True)
    state["completed_state_models_pruned"] = True
    state["resumable_model_state"] = "parent_kernel_state"
    save_state(state_path, state)


def reset_teacher_training(output_root, state):
    output_root = Path(output_root)
    models_root = output_root / "models"
    for pattern in ("teacher-*", "student-*"):
        for path in models_root.glob(pattern):
            shutil.rmtree(path, ignore_errors=True)
    for relative in ("oof", "selection", "final", "release"):
        shutil.rmtree(output_root / relative, ignore_errors=True)
    for relative in ("members.json", "medical-segmentation-v1.6-release.zip"):
        (output_root / relative).unlink(missing_ok=True)
    state["completed"] = [
        key
        for key in state.get("completed", [])
        if not key.startswith("model:teacher-")
        and not key.startswith("model:student-")
        and key not in {"oof_generated", "final_evaluation"}
    ]
    state["failed"] = {
        key: value
        for key, value in state.get("failed", {}).items()
        if not key.startswith("teacher-") and not key.startswith("student-")
    }
    state["phase"] = "teachers_unetpp"
    state["teacher_restart_reason"] = "reset_seed_epoch_and_optimizer_state"


def package_state(output_root):
    output_root = Path(output_root)
    _remove_recomputable_paths(output_root, RECOMPUTABLE_STATE_PATHS)
    prune_completed_pipeline_state(output_root)
    prune_redundant_checkpoints(output_root)
    archive = output_root.parent / "v1_6_state.zip"
    temporary_archive = archive.with_name(f".{archive.name}.tmp")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    temporary_checksum = checksum.with_name(f".{checksum.name}.tmp")
    archive.unlink(missing_ok=True)
    temporary_archive.unlink(missing_ok=True)
    checksum.unlink(missing_ok=True)
    temporary_checksum.unlink(missing_ok=True)
    excluded = {
        "prepared",
        "fold_data",
        "pretrain_data",
        "target_train_data",
        "validation_probability_cache",
        "streaming",
        "release",
        "candidates",
    }
    paths = [
        path
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path.name != "medical-segmentation-v1.6-release.zip"
        and not any(part in excluded for part in path.relative_to(output_root).parts)
    ]
    unpacked_bytes = sum(path.stat().st_size for path in paths)
    free_bytes = shutil.disk_usage(output_root.parent).free
    required_bytes = unpacked_bytes + STATE_PACKAGING_HEADROOM_BYTES
    print(
        f"State packaging input: {unpacked_bytes / (1024 ** 3):.2f} GiB; "
        f"free disk: {free_bytes / (1024 ** 3):.2f} GiB.",
        flush=True,
    )
    if free_bytes < required_bytes:
        raise OSError(
            "Insufficient disk for atomic state packaging after cleanup: "
            f"required={required_bytes}, free={free_bytes}"
        )
    try:
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as handle:
            for path in paths:
                handle.write(path, path.relative_to(output_root))
        os.replace(temporary_archive, archive)
        digest = sha256_file(archive)
        temporary_checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        os.replace(temporary_checksum, checksum)
    except BaseException:
        temporary_archive.unlink(missing_ok=True)
        temporary_checksum.unlink(missing_ok=True)
        raise
    print(f"State package: {archive} ({digest})", flush=True)
    return archive


class TimeBudget:
    def __init__(self, runtime_minutes, reserve_minutes):
        self.deadline = time.monotonic() + float(runtime_minutes) * 60
        self.reserve_seconds = float(reserve_minutes) * 60

    def can_start(self, minutes=8):
        return self.deadline - time.monotonic() - self.reserve_seconds >= float(minutes) * 60

    def remaining_minutes(self):
        return max(0.0, (self.deadline - time.monotonic() - self.reserve_seconds) / 60)


def _materialize_subset(images_dir, masks_dir, stems, output_root):
    images_dir, masks_dir, output_root = Path(images_dir), Path(masks_dir), Path(output_root)
    image_map = {path.stem: path for path in images_dir.iterdir() if path.is_file()}
    mask_map = {path.stem: path for path in masks_dir.iterdir() if path.is_file()}

    def resolve(stem):
        if stem in image_map and stem in mask_map:
            return stem
        # Legacy manifests may still carry the unprefixed original stem. Only
        # resolve it when exactly one materialized image/mask pair matches.
        candidates = [
            candidate
            for candidate in set(image_map) & set(mask_map)
            if candidate.partition("__")[2] == stem
        ]
        return candidates[0] if len(candidates) == 1 else None

    for split, ids in stems.items():
        image_out, mask_out = output_root / split / "images", output_root / split / "masks"
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        for stem in ids:
            resolved_stem = resolve(stem)
            if resolved_stem is None:
                raise ValueError(f"Subset references an unknown sample: {stem}")
            for source, target in ((image_map[resolved_stem], image_out / image_map[resolved_stem].name), (mask_map[resolved_stem], mask_out / mask_map[resolved_stem].name)):
                if not target.exists():
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)


def _materialize_ham_pretrain(images_dir, masks_dir, records, validation_groups, output_root):
    """Build HAM pretraining files from source pairs, not a manifest stem alias."""
    pair_map = {stem: (image_path, mask_path) for stem, image_path, mask_path in discover_pairs(images_dir, masks_dir)}
    accepted = [
        row
        for row in records
        if row.get("status") == "accepted" and row.get("source") == "ham10000"
    ]
    if not accepted:
        raise ValueError("No accepted HAM10000 records are available for pretraining.")
    validation_groups = {str(group) for group in validation_groups}
    counts = {"train": 0, "val": 0}
    output_root = Path(output_root)
    for row in sorted(accepted, key=lambda item: str(item["original_stem"])):
        original = str(row["original_stem"])
        pair = pair_map.get(original)
        if pair is None:
            raise ValueError(f"Accepted HAM10000 record has no source image/mask pair: {original}")
        split = "val" if str(row["group_id"]) in validation_groups else "train"
        image_path, mask_path = pair
        stem = materialized_stem("ham10000", original)
        image_target = output_root / split / "images" / f"{stem}{image_path.suffix.lower()}"
        mask_target = output_root / split / "masks" / f"{stem}{mask_path.suffix.lower()}"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        for source, target in ((image_path, image_target), (mask_path, mask_target)):
            if target.exists():
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        counts[split] += 1
    if not counts["train"] or not counts["val"]:
        raise ValueError("HAM10000 pretraining materialization requires both train and validation samples.")
    return counts


def prepare_data(args, output_root, state):
    output_root = Path(output_root)
    prepared = output_root / "prepared"
    merged = output_root / "merged_train"
    required = [
        merged / "images",
        merged / "masks",
        prepared / "internal/val/images",
        prepared / "internal/test/images",
        output_root / "pretrain_data/train/images",
        output_root / "pretrain_data/val/images",
        output_root / "target_train_data/train/images",
    ]
    was_completed = "data_prepared" in state["completed"]
    previous_phase = state.get("phase", "data")
    if was_completed and all(path.exists() for path in required):
        return merged
    for path in (prepared, merged, output_root / "fold_data", output_root / "pretrain_data", output_root / "target_train_data"):
        if path.exists():
            shutil.rmtree(path)
    prepare_internal_splits(args.isic17_root, prepared / "internal", image_size=None)
    prepare_external_split(args.isic18_root, prepared / "external", excluded_ids=collect_sample_ids(args.isic17_root), image_size=None)
    result = prepare_multisource_dataset(
        [
            ("isic17", prepared / "internal/train/images", prepared / "internal/train/masks"),
            ("isic16", args.isic16_images, args.isic16_masks),
            ("ph2", args.ph2_images, args.ph2_masks),
            ("ham10000", args.ham_images, args.ham_masks),
        ],
        [
            ("isic17_val", prepared / "internal/val/images", prepared / "internal/val/masks"),
            ("isic17_test", prepared / "internal/test/images", prepared / "internal/test/masks"),
            ("isic18_external", prepared / "external/images", prepared / "external/masks"),
        ],
        merged,
    )
    records = decorate_v16_manifest(result["manifest"], args.ham_metadata)
    folds = create_target_domain_folds(records, k=5, seed=42)
    folds_path = write_v16_folds(
        output_root / "target_folds.json",
        folds,
        metadata={"k": 5, "validation_source": "isic17", "stratification": "source|contrast_tertile|lesion_ratio_tertile"},
    )
    materialize_fold_directories(merged / "images", merged / "masks", folds, output_root / "fold_data", mode="hardlink")
    pretrain = create_ham_pretrain_split(records)
    ham_counts = _materialize_ham_pretrain(
        args.ham_images,
        args.ham_masks,
        records,
        pretrain["validation_groups"],
        output_root / "pretrain_data",
    )
    target_ids = [row["stem"] for row in records if row.get("status") == "accepted" and row.get("source") != "ham10000"]
    _materialize_subset(
        merged / "images",
        merged / "masks",
        {"train": target_ids},
        output_root / "target_train_data",
    )
    accepted = [row for row in records if row.get("status") == "accepted"]
    sources = {
        "ham10000": {
            "accepted": sum(row["source"] == "ham10000" for row in accepted),
            "label_quality": "reviewed",
            "license": HAM_LICENSE,
            "role": "pretrain_only",
            "materialized": ham_counts,
        },
        "isic17": {"accepted": sum(row["source"] == "isic17" for row in accepted), "role": "target_domain"},
        "isic16": {"accepted": sum(row["source"] == "isic16" for row in accepted), "role": "finetune"},
        "ph2": {"accepted": sum(row["source"] == "ph2" for row in accepted), "role": "finetune"},
    }
    (output_root / "data_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    manifest_hash = sha256_file(result["manifest"])
    previous_manifest = state.get("data_manifest_sha256")
    if previous_manifest not in {None, manifest_hash}:
        raise ValueError(f"Rematerialized data manifest differs from state: {previous_manifest} != {manifest_hash}")
    state.update(
        {
            "data_manifest_sha256": manifest_hash,
            "folds_sha256": sha256_file(folds_path),
            "phase": previous_phase if was_completed else "pretrain",
        }
    )
    if not was_completed:
        state["completed"].append("data_prepared")
    return merged


def prepare_evaluation_data(args, output_root):
    output_root = Path(output_root)
    prepared = output_root / "prepared"
    required = (
        prepared / "internal/val/images",
        prepared / "internal/val/masks",
        prepared / "internal/test/images",
        prepared / "internal/test/masks",
        prepared / "external/images",
        prepared / "external/masks",
    )
    if all(path.exists() for path in required):
        return prepared
    shutil.rmtree(prepared, ignore_errors=True)
    prepare_internal_splits(args.isic17_root, prepared / "internal", image_size=None)
    prepare_external_split(
        args.isic18_root,
        prepared / "external",
        excluded_ids=collect_sample_ids(args.isic17_root),
        image_size=None,
    )
    return prepared


def architecture_config(base, architecture):
    definition = ARCHITECTURES[architecture]
    config = copy.deepcopy(base)
    config["model"] = {"model_name": definition["model_name"], "encoder_name": definition["encoder_name"], "encoder_weights": "imagenet", "in_channels": 3, "out_channels": 1}
    config["training"]["batch_size"] = definition["batch_size"]
    config["training"]["gradient_accumulation_steps"] = definition["accumulation"]
    config["tracking"] = {"enabled": False, "mode": "disabled"}
    return config


def _finished(checkpoint, epochs):
    if not Path(checkpoint).exists():
        return False
    payload = load_checkpoint_payload(checkpoint, device="cpu")
    return int(payload.get("epoch", 0)) >= int(epochs) or payload.get("stopped_reason") == "early_stopping"


def train_stage(name, architecture, base, output_root, train_dirs, val_dirs, manifest, budget, state, epochs, phase, proportions=None, loss=None, soft_masks=None, seed_checkpoint=None, fixed_epochs=False):
    key = f"model:{name}:{phase}"
    task_root = Path(output_root) / "models" / name
    checkpoint_dir = task_root / "checkpoints"
    last_path, best_path = checkpoint_dir / "last_model.pth", checkpoint_dir / "best_model.pth"
    if key in state["completed"]:
        return True
    if not budget.can_start():
        return False
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seeded_from_checkpoint = False
    if seed_checkpoint and not last_path.exists():
        _seed_last, seed_best = (Path(item) for item in seed_checkpoint)
        if not _seed_last.exists() or not seed_best.exists():
            raise FileNotFoundError(f"Missing pretraining seed for {name}")
        shutil.copy2(seed_best, last_path)
        shutil.copy2(seed_best, best_path)
        seeded_from_checkpoint = True
    config = architecture_config(base, architecture)
    config["experiment_name"] = name
    config["data"].update({
        "train_images_dir": str(train_dirs[0]), "train_masks_dir": str(train_dirs[1]),
        "val_images_dir": str(val_dirs[0]), "val_masks_dir": str(val_dirs[1]),
        "source_sampling": {"manifest": str(manifest), "proportions": proportions} if proportions else {},
    })
    if soft_masks:
        config["data"].update({"soft_masks_dir": str(soft_masks), "missing_soft_mask_value": 0.5})
    config["training"].update({"epochs": int(epochs), "max_runtime_minutes": max(4.0, budget.remaining_minutes()), "safe_stop_minutes": 2.0})
    if seeded_from_checkpoint:
        config["training"].update({"resume_optimizer": False, "resume_training_state": False})
    if fixed_epochs:
        config["training"]["early_stopping"]["enabled"] = False
    config["paths"] = {"output_dir": str(task_root / "outputs"), "checkpoint_dir": str(checkpoint_dir)}
    if loss:
        config["loss"] = loss
    runtime_config = task_root / "runtime_config.yaml"
    write_yaml(runtime_config, config)
    command = [sys.executable, "train.py", "--config", runtime_config]
    if last_path.exists():
        command.extend(["--resume", last_path])
    try:
        run(command)
    except Exception as exc:  # noqa: BLE001
        state["failed"][f"{name}:{phase}"] = repr(exc)
        raise
    if _finished(last_path, epochs):
        state["completed"].append(key)
        state["failed"].pop(f"{name}:{phase}", None)
        return True
    return False


def run_pretraining(base, output_root, manifest, budget, state):
    data_root = Path(output_root) / "pretrain_data"
    for architecture in ARCHITECTURES:
        completed = train_stage(
            f"pretrain-{architecture}", architecture, base, output_root,
            (data_root / "train/images", data_root / "train/masks"),
            (data_root / "val/images", data_root / "val/masks"), manifest, budget, state, 25, "pretrain",
            loss={"name": "bootstrapped_bce_dice", "beta": 0.85, "bce_weight": 0.5, "dice_weight": 0.5},
        )
        save_state(Path(output_root) / "pipeline_state.json", state)
        if not completed:
            return False
    state["phase"] = "teachers_unetpp"
    return True


def run_teachers(base, output_root, manifest, budget, state, architecture):
    folds = read_folds(Path(output_root) / "target_folds.json")
    for fold in folds:
        name = f"teacher-{architecture}-fold{fold['fold']}"
        root = Path(output_root) / "models" / f"pretrain-{architecture}" / "checkpoints"
        fold_root = Path(output_root) / "fold_data" / f"fold_{fold['fold']}"
        mixed = train_stage(name, architecture, base, output_root, (fold_root / "train/images", fold_root / "train/masks"), (fold_root / "val/images", fold_root / "val/masks"), manifest, budget, state, 18, "mixed", MIXED_PROPORTIONS, {"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5}, seed_checkpoint=(root / "last_model.pth", root / "best_model.pth"))
        save_state(Path(output_root) / "pipeline_state.json", state)
        if not mixed:
            return False
        adapted = train_stage(name, architecture, base, output_root, (fold_root / "train/images", fold_root / "train/masks"), (fold_root / "val/images", fold_root / "val/masks"), manifest, budget, state, 30, "adapt", ADAPT_PROPORTIONS, {"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5})
        save_state(Path(output_root) / "pipeline_state.json", state)
        if not adapted:
            return False
    state["phase"] = "teachers_segformer" if architecture == "unetpp" else "students"
    return True


def _select_oof_candidate(candidate_roots, output_root):
    output_root = Path(output_root)
    baseline = json.loads((candidate_roots["none"] / "family_selection.json").read_text(encoding="utf-8"))
    selected_mode, selected = "none", baseline
    baseline_metrics = baseline["oof_metrics"]
    for mode in ("flip", "multiscale_flip"):
        candidate = json.loads((candidate_roots[mode] / "family_selection.json").read_text(encoding="utf-8"))
        metrics = candidate["oof_metrics"]
        if metrics["dice"] >= baseline_metrics["dice"] + 0.001 and metrics["boundary_f1"] >= baseline_metrics["boundary_f1"]:
            selected_mode, selected = mode, candidate
    selected["tta"] = selected_mode
    selected["postprocess"] = {"enabled": False, "min_component_area": 0, "fill_holes": False}
    root = candidate_roots[selected_mode]
    probabilities = {
        architecture: np.asarray(np.load(root / f"{architecture}_crossfit.npy", mmap_mode="r"), dtype=np.float32)
        for architecture in ARCHITECTURES
    }
    targets = np.asarray(np.load(root / "targets_384.npy", mmap_mode="r"), dtype=np.float32)
    weights = selected["family_weights"]
    combined = sum(float(weights[name]) * probabilities[name] for name in probabilities)
    base_metrics = macro_metrics(combined, targets, threshold=0.5)
    cleaned = postprocess_masks(combined, 0.5)
    cleaned_metrics = macro_metrics(cleaned * 0.999 + 0.0005, targets, threshold=0.5)
    if cleaned_metrics["dice"] >= base_metrics["dice"] + 0.001 and cleaned_metrics["boundary_f1"] >= base_metrics["boundary_f1"]:
        selected["postprocess"] = {"enabled": True, "min_component_area": 64, "fill_holes": True}
        selected["oof_metrics"] = cleaned_metrics
    soft_dir = output_root / "oof/soft_masks"
    shutil.rmtree(soft_dir, ignore_errors=True)
    stems = selected["stems"]
    resize_mode = selected["resize_mode"]
    image_paths = {path.stem: path for path in (output_root / "merged_train/images").iterdir() if path.is_file()}
    for index, stem in enumerate(stems):
        image = cv2.imread(str(image_paths.get(stem, "")), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot restore selected OOF soft mask for {stem}")
        weighted = sum(float(weights[name]) * probabilities[name][index, 0] for name in probabilities)
        write_soft_mask(soft_dir / f"{stem}.png", restore_probability(weighted, image.shape[:2], resize_mode=resize_mode))
    (output_root / "oof/family_selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def run_oof(output_root, state):
    if "oof_generated" in state["completed"]:
        return True
    output_root = Path(output_root)
    prune_runtime_data_for_oof(output_root)
    candidate_roots = {}
    baseline_metrics = None
    selected_mode = None
    for mode in ("none", "flip", "multiscale_flip"):
        candidate_root = output_root / "oof/candidates" / mode
        candidate_roots[mode] = candidate_root
        command = [
            sys.executable, "scripts/generate_crossfit_oof_v1_6.py", "--folds", output_root / "target_folds.json",
            "--manifest", output_root / "merged_train/data_manifest.csv", "--images-dir", output_root / "merged_train/images",
            "--masks-dir", output_root / "merged_train/masks", "--output-root", candidate_root, "--tta", mode,
        ]
        for architecture in ARCHITECTURES:
            for fold in range(5):
                root = output_root / "models" / f"teacher-{architecture}-fold{fold}"
                checkpoint = root / "checkpoints/best_model.pth"
                command.extend(["--member", f"{architecture}:{fold}:{root / 'runtime_config.yaml'}:{checkpoint}"])
        run(command)
        decision = json.loads((candidate_root / "family_selection.json").read_text(encoding="utf-8"))
        if mode == "none":
            baseline_metrics = decision["oof_metrics"]
            selected_mode = mode
            continue
        metrics = decision["oof_metrics"]
        accepted = (
            metrics["dice"] >= baseline_metrics["dice"] + 0.001
            and metrics["boundary_f1"] >= baseline_metrics["boundary_f1"]
        )
        if accepted:
            _prune_oof_candidate_payload(candidate_roots[selected_mode])
            selected_mode = mode
        else:
            _prune_oof_candidate_payload(candidate_root)
    _select_oof_candidate(candidate_roots, output_root)
    shutil.rmtree(output_root / "oof/candidates", ignore_errors=True)
    state["completed"].append("oof_generated")
    return True


def _median_epoch(output_root, architecture):
    values = []
    for fold in range(5):
        checkpoint = Path(output_root) / "models" / f"teacher-{architecture}-fold{fold}/checkpoints/best_model.pth"
        values.append(int(load_checkpoint_payload(checkpoint, device="cpu").get("epoch", 30)))
    return sorted(values)[len(values) // 2]


def run_students(base, output_root, manifest, budget, state):
    output_root = Path(output_root)
    if not run_oof(output_root, state):
        return False
    train_dirs = (output_root / "target_train_data/train/images", output_root / "target_train_data/train/masks")
    # HAM validation is held out from the student target-domain train pool and cannot leak ISIC17 validation.
    val_dirs = (output_root / "pretrain_data/val/images", output_root / "pretrain_data/val/masks")
    for architecture in ARCHITECTURES:
        name = f"student-{architecture}"
        seed_root = output_root / "models" / f"pretrain-{architecture}" / "checkpoints"
        total_epochs = max(12, _median_epoch(output_root, architecture))
        completed = train_stage(
            name, architecture, base, output_root, train_dirs, val_dirs, manifest, budget, state, total_epochs, "student",
            ADAPT_PROPORTIONS,
            {"name": "confidence_gated_distillation", "hard_weight": 0.90, "soft_bce_weight": 0.07, "soft_dice_weight": 0.03, "temperature": 2.0, "confidence_threshold": 0.60, "hard_loss": {"bce_weight": 0.5, "dice_weight": 0.5}},
            soft_masks=output_root / "oof/soft_masks", seed_checkpoint=(seed_root / "last_model.pth", seed_root / "best_model.pth"), fixed_epochs=True,
        )
        save_state(output_root / "pipeline_state.json", state)
        if not completed:
            return False
    state["phase"] = "selection"
    return True


def build_members(output_root):
    output_root = Path(output_root)
    members = []
    for architecture in ARCHITECTURES:
        for fold in range(5):
            root = output_root / "models" / f"teacher-{architecture}-fold{fold}"
            members.append({"name": f"teacher-{architecture}-fold{fold}", "kind": "teacher", "architecture": architecture, "config": str(root / "runtime_config.yaml"), "checkpoint": str(root / "checkpoints/best_model.pth")})
        root = output_root / "models" / f"student-{architecture}"
        members.append({"name": f"student-{architecture}", "kind": "student", "architecture": architecture, "config": str(root / "runtime_config.yaml"), "checkpoint": str(root / "checkpoints/last_model.pth")})
    path = output_root / "members.json"
    path.write_text(json.dumps(members, indent=2), encoding="utf-8")
    return path


def package_release(output_root, members_path, decision_path):
    output_root = Path(output_root)
    result_path = output_root / "final/evaluation_complete.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    members = {item["name"]: item for item in json.loads(Path(members_path).read_text(encoding="utf-8"))}
    decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    release = output_root / "release"
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)
    published = []
    if results["publish_default"]:
        spec = members[decision["fast"]["member"]]
        target = release / "fast"
        target.mkdir()
        shutil.copy2(spec["checkpoint"], target / "best_model.pth")
        shutil.copy2(spec["config"], target / "runtime_config.yaml")
        (target / "decision.json").write_text(json.dumps(decision["fast"], indent=2), encoding="utf-8")
        published.append("fast")
    if results["publish_best_accuracy"]:
        target = release / "best_accuracy"
        for name in decision["members"]:
            spec = members[name]
            member = target / "members" / name
            member.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spec["checkpoint"], member / "best_model.pth")
            shutil.copy2(spec["config"], member / "runtime_config.yaml")
        (target / "ensemble.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        published.append("best_accuracy")
    for path in [decision_path, result_path, output_root / "data_sources.json"]:
        if Path(path).exists():
            shutil.copy2(path, release / Path(path).name)
    data_manifest = output_root / "merged_train/data_manifest.csv"
    if data_manifest.exists():
        shutil.copy2(data_manifest, release / data_manifest.name)
    else:
        state = load_state(output_root / "pipeline_state.json")
        provenance = {
            "data_manifest_sha256": state.get("data_manifest_sha256"),
            "folds_sha256": state.get("folds_sha256"),
            "source_commit": state.get("source_commit"),
            "manifest_retained_in_parent_state": True,
        }
        (release / "training_data_provenance.json").write_text(
            json.dumps(provenance, indent=2),
            encoding="utf-8",
        )
    manifest = {"version": "1.6", "published_variants": published, "default_replaced": results["publish_default"], "files": {}}
    for path in sorted(release.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(release))] = sha256_file(path)
    (release / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = output_root / "medical-segmentation-v1.6-release.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as handle:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(release))
    return archive


def run_selection_and_evaluation(output_root, state):
    output_root = Path(output_root)
    members = build_members(output_root)
    selection = output_root / "selection"
    decision = selection / "locked_decision.json"
    if not decision.exists():
        run([sys.executable, "scripts/select_ensemble_v1_6.py", "--members-json", members, "--family-selection", output_root / "oof/family_selection.json", "--images-dir", output_root / "prepared/internal/val/images", "--masks-dir", output_root / "prepared/internal/val/masks", "--output-root", selection])
    test_manifest = output_root / "prepared/internal/test_manifest.csv"
    if not test_manifest.exists():
        run([sys.executable, "scripts/create_split_manifest.py", "--images-dir", output_root / "prepared/internal/test/images", "--masks-dir", output_root / "prepared/internal/test/masks", "--output", test_manifest, "--source", "isic17_test"])
    final = output_root / "final/evaluation_complete.json"
    if not final.exists():
        run([sys.executable, "scripts/evaluate_locked_v1_6.py", "--members-json", members, "--decision", decision, "--test-images", output_root / "prepared/internal/test/images", "--test-masks", output_root / "prepared/internal/test/masks", "--external-images", output_root / "prepared/external/images", "--external-masks", output_root / "prepared/external/masks", "--test-manifest", test_manifest, "--output-root", output_root / "final"])
    package_release(output_root, members, decision)
    state["phase"] = "complete"
    state["completed"].append("final_evaluation")


def main():
    parser = argparse.ArgumentParser(description="Run or resume the v1.6 five-stage Kaggle pipeline.")
    parser.add_argument("--config", default="configs/kaggle_v1_6.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-input")
    parser.add_argument("--isic17-root", required=True)
    parser.add_argument("--isic18-root", required=True)
    parser.add_argument("--isic16-images", required=True)
    parser.add_argument("--isic16-masks", required=True)
    parser.add_argument("--ph2-images", required=True)
    parser.add_argument("--ph2-masks", required=True)
    parser.add_argument("--ham-images", required=True)
    parser.add_argument("--ham-masks", required=True)
    parser.add_argument("--ham-metadata", required=True)
    parser.add_argument("--runtime-minutes", type=float, default=570)
    parser.add_argument("--reserve-minutes", type=float, default=30)
    parser.add_argument("--allow-state-mismatch", action="store_true")
    parser.add_argument("--reset-teachers", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    restore_state_archive(args.state_input, output_root)
    state_path = output_root / "pipeline_state.json"
    state = load_state(state_path)
    base = load_config(args.config)
    config_hash = sha256_file(args.config)
    source_commit = git_commit()
    if not args.allow_state_mismatch:
        if state.get("config_sha256") not in {None, config_hash}:
            raise ValueError("State config hash does not match the v1.6 config.")
        if state.get("source_commit") not in {None, source_commit}:
            raise ValueError(f"State commit mismatch: {state['source_commit']} != {source_commit}")
    state.update(
        {
            "version": "1.6",
            "config_sha256": config_hash,
            "source_commit": source_commit,
            "session": int(state.get("session", 0)) + 1,
        }
    )
    if args.reset_teachers:
        reset_teacher_training(output_root, state)
    budget = TimeBudget(args.runtime_minutes, args.reserve_minutes)
    try:
        if state.get("phase") == "selection":
            prepare_evaluation_data(args, output_root)
            manifest = None
        else:
            merged = prepare_data(args, output_root, state)
            manifest = merged / "data_manifest.csv"
            prune_runtime_data_for_phase(output_root, state.get("phase"))
        if args.prepare_only:
            save_state(state_path, state)
        elif state["phase"] == "pretrain" and budget.can_start():
            run_pretraining(base, output_root, manifest, budget, state)
        elif state["phase"] == "teachers_unetpp" and budget.can_start():
            run_teachers(base, output_root, manifest, budget, state, "unetpp")
        elif state["phase"] == "teachers_segformer" and budget.can_start():
            run_teachers(base, output_root, manifest, budget, state, "segformer")
        elif state["phase"] == "students" and budget.can_start():
            run_students(base, output_root, manifest, budget, state)
        elif state["phase"] == "selection" and budget.can_start(20):
            run_selection_and_evaluation(output_root, state)
        save_state(state_path, state)
    finally:
        save_state(state_path, state)
        package_state(output_root)
    print(json.dumps({"phase": state["phase"], "session": state["session"]}, indent=2))


if __name__ == "__main__":
    main()
