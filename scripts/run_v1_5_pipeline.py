import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cross_validation import (  # noqa: E402
    create_stratified_kfold_splits,
    materialize_fold_directories,
    read_folds,
    write_folds,
)
from src.data_preparation import collect_sample_ids, prepare_external_split, prepare_internal_splits  # noqa: E402
from src.multisource_data import prepare_multisource_dataset, read_manifest  # noqa: E402
from src.tracking import create_tracker  # noqa: E402
from src.utils import load_checkpoint_payload, load_config  # noqa: E402

ARCHITECTURES = {
    "unetpp": {"model_name": "unet_plus_plus", "encoder_name": "efficientnet-b3", "batch_size": 6},
    "segformer": {"model_name": "segformer", "encoder_name": "nvidia/mit-b2", "batch_size": 4},
    "manet": {"model_name": "manet", "encoder_name": "efficientnet-b3", "batch_size": 6},
}


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
        return {"version": "1.5", "session": 0, "phase": "data", "completed": [], "failed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def log_pipeline_snapshot(tracker, output_root, state):
    output_root = Path(output_root)
    values = {
        "pipeline/session": state.get("session", 0),
        "pipeline/phase": state.get("phase", "unknown"),
        "pipeline/completed_tasks": len(state.get("completed", [])),
        "pipeline/teacher_folds_completed": sum(
            item.startswith("model:teacher-") for item in state.get("completed", [])
        ),
    }
    manifest_path = output_root / "merged_train/data_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sources = sorted({row["source"] for row in rows})
        for source in sources:
            values[f"data/{source}_accepted"] = sum(
                row["source"] == source and row["status"] == "accepted" for row in rows
            )
            values[f"data/{source}_removed_duplicates"] = sum(
                row["source"] == source and row["status"] == "removed_duplicate" for row in rows
            )
    tracker.log(values)
    oof_path = output_root / "oof/hard_example_weights.csv"
    if oof_path.exists():
        with oof_path.open(newline="", encoding="utf-8") as handle:
            oof_rows = list(csv.DictReader(handle))
        tracker.log_histogram("oof/dice_distribution", [float(row["dice"]) for row in oof_rows])
        tracker.log_histogram("oof/boundary_f1_distribution", [float(row["boundary_f1"]) for row in oof_rows])
        tracker.log_histogram("oof/hard_example_weights", [float(row["weight"]) for row in oof_rows])
    selection_path = output_root / "selection/selection_history.csv"
    if selection_path.exists():
        with selection_path.open(newline="", encoding="utf-8") as handle:
            selection = list(csv.DictReader(handle))
        columns = ["step", "member", "improvement", "dice", "boundary_f1", "composite", "threshold"]
        tracker.log_table(
            "ensemble/member_selection",
            columns,
            [[row.get(column) for column in columns] for row in selection],
        )
    final_path = output_root / "final/evaluation_complete.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        rows = []
        for variant in ["fast", "best_accuracy"]:
            for split in ["test", "external"]:
                metrics = final[variant][split]
                rows.append(
                    [
                        variant,
                        split,
                        metrics.get("dice"),
                        metrics.get("boundary_f1"),
                        metrics.get("low_contrast_dice"),
                        metrics.get("accepted"),
                    ]
                )
        tracker.log_table(
            "final/fast_vs_best_accuracy",
            ["variant", "split", "dice", "boundary_f1", "low_contrast_dice", "accepted"],
            rows,
        )


def restore_state_archive(archive, output_root):
    if not archive:
        return
    archive = Path(archive)
    if not archive.exists():
        raise FileNotFoundError(f"State archive does not exist: {archive}")
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(archive)
        if actual != expected:
            raise ValueError(f"State archive SHA256 mismatch: expected={expected}, actual={actual}")
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(output_root)


def package_state(output_root):
    output_root = Path(output_root)
    archive = output_root.parent / "v1_5_state.zip"
    excluded_parts = {"validation_probability_cache", "streaming", "fold_data", "release"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as handle:
        for path in sorted(output_root.rglob("*")):
            if not path.is_file() or any(part in excluded_parts for part in path.parts):
                continue
            relative = path.relative_to(output_root)
            if relative.name == "medical-segmentation-v1.5-release.zip":
                continue
            if relative.parts[0] == "prepared" and relative.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                continue
            if relative.parts[:2] in {("merged_train", "images"), ("merged_train", "masks")}:
                continue
            handle.write(path, relative)
    digest = sha256_file(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"State package: {archive} ({digest})", flush=True)
    return archive


class TimeBudget:
    def __init__(self, runtime_minutes, reserve_minutes):
        self.started = time.time()
        self.deadline = self.started + float(runtime_minutes) * 60.0
        self.reserve_seconds = float(reserve_minutes) * 60.0

    @property
    def usable_seconds(self):
        return max(0.0, self.deadline - time.time() - self.reserve_seconds)

    def can_start(self, minimum_minutes=8):
        return self.usable_seconds >= float(minimum_minutes) * 60.0


def prepare_data(args, output_root, state):
    prepared_root = output_root / "prepared"
    merged_root = output_root / "merged_train"
    was_completed = "data_prepared" in state["completed"]
    materialized = (
        (merged_root / "images").exists()
        and (merged_root / "masks").exists()
        and (prepared_root / "internal/val/images").exists()
        and (prepared_root / "internal/test/images").exists()
        and (prepared_root / "external/images").exists()
    )
    if was_completed and materialized:
        fold_data = output_root / "fold_data"
        if not fold_data.exists():
            materialize_fold_directories(
                merged_root / "images",
                merged_root / "masks",
                read_folds(output_root / "folds.json"),
                fold_data,
                mode="hardlink",
            )
        return merged_root
    previous_phase = state.get("phase", "data")
    expected_manifest_hash = state.get("data_manifest_sha256")
    expected_folds_hash = state.get("folds_sha256")
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    if merged_root.exists():
        shutil.rmtree(merged_root)
    fold_data = output_root / "fold_data"
    if fold_data.exists():
        shutil.rmtree(fold_data)
    internal_report = prepare_internal_splits(args.isic17_root, prepared_root / "internal", image_size=None)
    external_report = prepare_external_split(
        args.isic18_root,
        prepared_root / "external",
        excluded_ids=collect_sample_ids(args.isic17_root),
        image_size=None,
    )
    result = prepare_multisource_dataset(
        sources=[
            ("isic17", prepared_root / "internal/train/images", prepared_root / "internal/train/masks"),
            ("isic16", args.isic16_images, args.isic16_masks),
            ("ph2", args.ph2_images, args.ph2_masks),
        ],
        benchmarks=[
            ("isic17_val", prepared_root / "internal/val/images", prepared_root / "internal/val/masks"),
            ("isic17_test", prepared_root / "internal/test/images", prepared_root / "internal/test/masks"),
            ("isic18_external", prepared_root / "external/images", prepared_root / "external/masks"),
        ],
        output_root=merged_root,
    )
    accepted_by_source = {}
    removed_by_source = {}
    for row in result["rows"]:
        target = accepted_by_source if row["status"] == "accepted" else removed_by_source
        target[row["source"]] = target.get(row["source"], 0) + 1
    sources = {
        "isic17": {
            "candidates": internal_report["prepared_splits"]["train"],
            "accepted": accepted_by_source.get("isic17", 0),
            "removed_duplicates": removed_by_source.get("isic17", 0),
            "license": "dataset-provided terms",
        },
        "isic16": {
            "accepted": accepted_by_source.get("isic16", 0),
            "removed_duplicates": removed_by_source.get("isic16", 0),
            "license": "CC0",
        },
        "ph2": {
            "accepted": accepted_by_source.get("ph2", 0),
            "removed_duplicates": removed_by_source.get("ph2", 0),
            "license": "research use; cite PH2 publication",
        },
        "isic18_external": {"samples": external_report["prepared_pairs"], "role": "external evaluation only"},
        "accepted_after_dedup": result["accepted"],
    }
    (output_root / "data_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    records = read_manifest(result["manifest"])
    folds = create_stratified_kfold_splits(records, k=5, seed=42)
    folds_path = write_folds(
        output_root / "folds.json",
        folds,
        metadata={"k": 5, "seed": 42, "stratification": "source|contrast_tertile|lesion_ratio_tertile"},
    )
    materialize_fold_directories(
        merged_root / "images", merged_root / "masks", folds, output_root / "fold_data", mode="hardlink"
    )
    actual_manifest_hash = sha256_file(result["manifest"])
    actual_folds_hash = sha256_file(folds_path)
    if expected_manifest_hash not in {None, actual_manifest_hash}:
        raise ValueError(
            f"Rematerialized data manifest differs from saved state: {expected_manifest_hash} != {actual_manifest_hash}"
        )
    if expected_folds_hash not in {None, actual_folds_hash}:
        raise ValueError(f"Rematerialized folds differ from saved state: {expected_folds_hash} != {actual_folds_hash}")
    state.update(
        {
            "phase": previous_phase if was_completed else "screening",
            "data_manifest_sha256": actual_manifest_hash,
            "folds_sha256": actual_folds_hash,
        }
    )
    if not was_completed:
        state["completed"].append("data_prepared")
    return merged_root


def architecture_config(base, architecture):
    definition = ARCHITECTURES[architecture]
    config = copy.deepcopy(base)
    config["model"] = {
        "model_name": definition["model_name"],
        "in_channels": 3,
        "out_channels": 1,
        "encoder_name": definition["encoder_name"],
        "encoder_weights": "imagenet",
    }
    config["training"]["batch_size"] = definition["batch_size"]
    return config


def task_complete(task_root, epochs):
    last_path = Path(task_root) / "checkpoints/last_model.pth"
    if not last_path.exists():
        return False
    payload = load_checkpoint_payload(last_path, device="cpu")
    return int(payload.get("epoch", 0)) >= int(epochs) or payload.get("stopped_reason") == "early_stopping"


def train_task(
    name,
    architecture,
    base_config,
    train_dirs,
    val_dirs,
    output_root,
    group,
    budget,
    manifest,
    epochs,
    patience,
    job_type,
    state,
    resume_seed=None,
    distillation=None,
):
    task_root = output_root / "models" / name
    task_key = f"model:{name}"
    if task_key in state["completed"]:
        return True
    if not budget.can_start():
        return False
    task_root.mkdir(parents=True, exist_ok=True)
    state.setdefault("wandb_runs", {})[name] = name
    last_path = task_root / "checkpoints/last_model.pth"
    best_path = task_root / "checkpoints/best_model.pth"
    if not last_path.exists() and resume_seed:
        seed_last, seed_best = map(Path, resume_seed)
        if not seed_last.exists() or not seed_best.exists():
            raise FileNotFoundError(f"Screening seed checkpoints are incomplete: last={seed_last}, best={seed_best}")
        last_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_last, last_path)
        shutil.copy2(seed_best, best_path)
    resumed_run_id = None
    resumed_experiment = None
    resumed_epoch = 0
    resume_payload = None
    if last_path.exists():
        resume_payload = load_checkpoint_payload(last_path, device="cpu")
        resumed_run_id = resume_payload.get("metadata", {}).get("wandb_run_id")
        resumed_experiment = resume_payload.get("config", {}).get("experiment_name")
        resumed_epoch = int(resume_payload.get("epoch", 0))
    seed_run_id = f"screen-{architecture}-fold0"
    is_screen_transition = bool(resume_seed and seed_run_id in {resumed_run_id, resumed_experiment})
    config = architecture_config(base_config, architecture)
    config["experiment_name"] = name
    config["data"].update(
        {
            "train_images_dir": str(train_dirs[0]),
            "train_masks_dir": str(train_dirs[1]),
            "val_images_dir": str(val_dirs[0]),
            "val_masks_dir": str(val_dirs[1]),
        }
    )
    config["data"].setdefault("source_sampling", {})["manifest"] = str(manifest)
    config["training"]["epochs"] = int(epochs)
    config["training"]["max_runtime_minutes"] = max(4.0, budget.usable_seconds / 60.0)
    config["training"]["safe_stop_minutes"] = 2.0
    config["training"]["early_stopping"]["patience"] = int(patience)
    if is_screen_transition:
        config["training"]["resume_optimizer"] = False
        config["training"]["scheduler_epochs"] = max(1, int(epochs) - resumed_epoch)
    elif resume_payload is not None:
        saved_scheduler_epochs = resume_payload.get("config", {}).get("training", {}).get("scheduler_epochs")
        if saved_scheduler_epochs is not None:
            config["training"]["scheduler_epochs"] = int(saved_scheduler_epochs)
    config["paths"] = {
        "output_dir": str(task_root / "outputs"),
        "checkpoint_dir": str(task_root / "checkpoints"),
    }
    runtime_config = task_root / "runtime_config.yaml"
    config["tracking"].update(
        {
            "run_id": name,
            "name": name,
            "group": group,
            "job_type": job_type,
            "offline_dir": str(output_root / "wandb-offline"),
            "artifact_files": [str(runtime_config), str(manifest), str(output_root / "data_sources.json")],
        }
    )
    if resume_seed:
        config["tracking"]["resume_source_run_ids"] = [seed_run_id]
    if distillation:
        config["data"]["soft_masks_dir"] = str(distillation["soft_masks_dir"])
        config["data"]["sample_weights_csv"] = str(distillation["weights"])
        config["loss"] = {
            "name": "distillation",
            "hard_weight": 0.60,
            "soft_bce_weight": 0.25,
            "soft_dice_weight": 0.15,
            "temperature": 2.0,
        }
    write_yaml(runtime_config, config)
    del resume_payload
    gc.collect()
    command = [sys.executable, "train.py", "--config", runtime_config]
    if last_path.exists():
        command.extend(["--resume", last_path])
    try:
        run(command)
    except Exception as exc:  # noqa: BLE001
        state["failed"][name] = repr(exc)
        save_state(output_root / "pipeline_state.json", state)
        if architecture == "segformer":
            print(f"SegFormer failed and will be excluded from screening: {exc}", flush=True)
            return False
        raise
    if task_complete(task_root, epochs):
        state["completed"].append(task_key)
        state["failed"].pop(name, None)
        return True
    return False


def checkpoint_composite(path):
    payload = load_checkpoint_payload(path, device="cpu")
    metrics = payload.get("best_metrics") or payload.get("val_metrics") or {}
    return float(metrics.get("composite", 0.75 * metrics.get("dice", 0.0) + 0.25 * metrics.get("boundary_f1", 0.0)))


def run_screening(base, output_root, group, budget, manifest, state):
    fold_root = output_root / "fold_data/fold_0"
    for architecture in ARCHITECTURES:
        name = f"screen-{architecture}-fold0"
        if name in state.get("failed", {}) and architecture == "segformer":
            continue
        train_task(
            name,
            architecture,
            base,
            (fold_root / "train/images", fold_root / "train/masks"),
            (fold_root / "val/images", fold_root / "val/masks"),
            output_root,
            group,
            budget,
            manifest,
            12,
            4,
            "screening",
            state,
        )
        save_state(output_root / "pipeline_state.json", state)
        if not budget.can_start():
            return False
    available = []
    for architecture in ARCHITECTURES:
        checkpoint = output_root / f"models/screen-{architecture}-fold0/checkpoints/best_model.pth"
        if checkpoint.exists() and f"model:screen-{architecture}-fold0" in state["completed"]:
            available.append((checkpoint_composite(checkpoint), architecture))
    if len(available) < 2:
        return False
    winners = [architecture for _, architecture in sorted(available, reverse=True)[:2]]
    state["architectures"] = winners
    state["phase"] = "teachers"
    return True


def run_teachers(base, output_root, group, budget, manifest, state):
    for architecture in state["architectures"]:
        for fold in range(5):
            name = f"teacher-{architecture}-fold{fold}"
            fold_root = output_root / f"fold_data/fold_{fold}"
            seed = None
            if fold == 0:
                screen_root = output_root / f"models/screen-{architecture}-fold0/checkpoints"
                seed = (screen_root / "last_model.pth", screen_root / "best_model.pth")
            completed = train_task(
                name,
                architecture,
                base,
                (fold_root / "train/images", fold_root / "train/masks"),
                (fold_root / "val/images", fold_root / "val/masks"),
                output_root,
                group,
                budget,
                manifest,
                35,
                7,
                "teacher",
                state,
                resume_seed=seed,
            )
            save_state(output_root / "pipeline_state.json", state)
            if not completed or not budget.can_start():
                return False
    state["phase"] = "oof"
    return True


def run_oof(output_root, state):
    if "oof_generated" in state["completed"]:
        return True
    command = [
        sys.executable,
        "scripts/generate_oof_targets.py",
        "--folds",
        output_root / "folds.json",
        "--manifest",
        output_root / "merged_train/data_manifest.csv",
        "--images-dir",
        output_root / "merged_train/images",
        "--masks-dir",
        output_root / "merged_train/masks",
        "--output-root",
        output_root / "oof",
    ]
    for architecture in state["architectures"]:
        for fold in range(5):
            task_root = output_root / f"models/teacher-{architecture}-fold{fold}"
            checkpoint = task_root / "checkpoints/best_model.pth"
            if not checkpoint.exists():
                checkpoint = task_root / "checkpoints/last_model.pth"
            command.extend(
                ["--member", f"{architecture}:{fold}:{task_root / 'runtime_config.yaml'}:{checkpoint}"]
            )
    run(command)
    state["completed"].append("oof_generated")
    state["phase"] = "students"
    return True


def run_students(base, output_root, group, budget, manifest, state):
    train_dirs = (output_root / "merged_train/images", output_root / "merged_train/masks")
    val_dirs = (output_root / "prepared/internal/val/images", output_root / "prepared/internal/val/masks")
    distillation = {"soft_masks_dir": output_root / "oof/soft_masks", "weights": output_root / "oof/hard_example_weights.csv"}
    for architecture in state["architectures"]:
        completed = train_task(
            f"student-{architecture}",
            architecture,
            base,
            train_dirs,
            val_dirs,
            output_root,
            group,
            budget,
            manifest,
            40,
            8,
            "student",
            state,
            distillation=distillation,
        )
        save_state(output_root / "pipeline_state.json", state)
        if not completed or not budget.can_start():
            return False
    state["phase"] = "selection"
    return True


def build_members(output_root, state):
    members = []
    for kind, folds in (("teacher", range(5)), ("student", [None])):
        for architecture in state["architectures"]:
            for fold in folds:
                name = f"{kind}-{architecture}" + (f"-fold{fold}" if fold is not None else "")
                root = output_root / "models" / name
                checkpoint = root / "checkpoints/best_model.pth"
                if not checkpoint.exists():
                    checkpoint = root / "checkpoints/last_model.pth"
                members.append(
                    {"name": name, "kind": kind, "architecture": architecture, "config": str(root / "runtime_config.yaml"), "checkpoint": str(checkpoint)}
                )
    path = output_root / "members.json"
    path.write_text(json.dumps(members, indent=2), encoding="utf-8")
    return path


def package_release(output_root, members_path, decision_path):
    output_root = Path(output_root)
    release_root = output_root / "release"
    if release_root.exists():
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True)
    members = {item["name"]: item for item in json.loads(Path(members_path).read_text(encoding="utf-8"))}
    decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    results = json.loads((output_root / "final/evaluation_complete.json").read_text(encoding="utf-8"))
    published = []
    if results.get("publish_default"):
        fast = decision["fast"]
        member = members[fast["member"]]
        target = release_root / "fast"
        target.mkdir()
        shutil.copy2(member["checkpoint"], target / "best_model.pth")
        config = load_config(member["config"])
        config.setdefault("inference", {})["threshold"] = fast["threshold"]
        write_yaml(target / "runtime_config.yaml", config)
        (target / "decision.json").write_text(json.dumps(fast, indent=2), encoding="utf-8")
        published.append("fast")
    if results.get("publish_best_accuracy"):
        target = release_root / "best_accuracy"
        target.mkdir()
        for name in decision["members"]:
            member_target = target / "members" / name
            member_target.mkdir(parents=True)
            shutil.copy2(members[name]["checkpoint"], member_target / "best_model.pth")
            shutil.copy2(members[name]["config"], member_target / "runtime_config.yaml")
        (target / "ensemble.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        published.append("best_accuracy")
    for path in [
        decision_path,
        output_root / "final/evaluation_complete.json",
        output_root / "merged_train/data_manifest.csv",
        output_root / "data_sources.json",
    ]:
        shutil.copy2(path, release_root / Path(path).name)
    manifest = {
        "version": "1.5",
        "published_variants": published,
        "default_replaced": results.get("publish_default", False),
        "files": {},
    }
    for path in sorted(release_root.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(release_root))] = sha256_file(path)
    (release_root / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = output_root / "medical-segmentation-v1.5-release.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as handle:
        for path in sorted(release_root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(release_root))
    return archive


def run_selection_and_evaluation(output_root, state):
    members = build_members(output_root, state)
    selection_root = output_root / "selection"
    decision = selection_root / "locked_decision.json"
    if not decision.exists():
        run(
            [
                sys.executable,
                "scripts/select_ensemble_v1_5.py",
                "--members-json",
                members,
                "--images-dir",
                output_root / "prepared/internal/val/images",
                "--masks-dir",
                output_root / "prepared/internal/val/masks",
                "--output-root",
                selection_root,
            ]
        )
    test_manifest = output_root / "prepared/internal/test_manifest.csv"
    if not test_manifest.exists():
        run(
            [
                sys.executable,
                "scripts/create_split_manifest.py",
                "--images-dir",
                output_root / "prepared/internal/test/images",
                "--masks-dir",
                output_root / "prepared/internal/test/masks",
                "--output",
                test_manifest,
                "--source",
                "isic17_test",
            ]
        )
    if not (output_root / "final/evaluation_complete.json").exists():
        run(
            [
                sys.executable,
                "scripts/evaluate_locked_v1_5.py",
                "--members-json",
                members,
                "--decision",
                decision,
                "--test-images",
                output_root / "prepared/internal/test/images",
                "--test-masks",
                output_root / "prepared/internal/test/masks",
                "--external-images",
                output_root / "prepared/external/images",
                "--external-masks",
                output_root / "prepared/external/masks",
                "--test-manifest",
                test_manifest,
                "--output-root",
                output_root / "final",
            ]
        )
    package_release(output_root, members, decision)
    state["phase"] = "complete"
    state["completed"].append("final_evaluation")


def main():
    parser = argparse.ArgumentParser(description="Run or resume the three-session v1.5 Kaggle pipeline.")
    parser.add_argument("--config", default="configs/kaggle_v1_5.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-input")
    parser.add_argument("--isic17-root", required=True)
    parser.add_argument("--isic18-root", required=True)
    parser.add_argument("--isic16-images", required=True)
    parser.add_argument("--isic16-masks", required=True)
    parser.add_argument("--ph2-images", required=True)
    parser.add_argument("--ph2-masks", required=True)
    parser.add_argument("--runtime-minutes", type=float, default=570)
    parser.add_argument("--reserve-minutes", type=float, default=30)
    parser.add_argument("--allow-state-mismatch", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    restore_state_archive(args.state_input, output_root)
    offline_root = output_root / "wandb-offline"
    if os.environ.get("WANDB_API_KEY") and offline_root.exists():
        offline_runs = list(offline_root.rglob("offline-run-*"))
        if offline_runs:
            try:
                run([sys.executable, "-m", "wandb", "sync", *offline_runs])
            except Exception as exc:  # noqa: BLE001
                print(f"W&B offline sync deferred: {exc}", flush=True)
    state_path = output_root / "pipeline_state.json"
    state = load_state(state_path)
    base = load_config(args.config)
    current_commit = git_commit()
    config_hash = sha256_file(args.config)
    if not args.allow_state_mismatch:
        if state.get("source_commit") not in {None, current_commit}:
            raise ValueError(f"State commit mismatch: {state.get('source_commit')} != {current_commit}")
        if state.get("config_sha256") not in {None, config_hash}:
            raise ValueError("State base config hash does not match the current v1.5 config.")
    state["source_commit"] = current_commit
    state["config_sha256"] = config_hash
    state["session"] = int(state.get("session", 0)) + 1
    budget = TimeBudget(args.runtime_minutes, args.reserve_minutes)

    try:
        previous_data_hash = state.get("data_manifest_sha256")
        merged_root = prepare_data(args, output_root, state)
        manifest = merged_root / "data_manifest.csv"
        actual_data_hash = sha256_file(manifest)
        if not args.allow_state_mismatch and previous_data_hash not in {None, actual_data_hash}:
            raise ValueError(f"State data manifest mismatch: {previous_data_hash} != {actual_data_hash}")
        state["data_manifest_sha256"] = actual_data_hash
        group = f"v1.5-{state['data_manifest_sha256'][:8]}"
        controller_config = copy.deepcopy(base)
        controller_config["experiment_name"] = "pipeline-controller"
        controller_config["tracking"].update(
            {
                "run_id": "pipeline-controller",
                "name": "pipeline-controller",
                "group": group,
                "job_type": "pipeline",
                "offline_dir": str(output_root / "wandb-offline"),
            }
        )
        controller = create_tracker(controller_config, output_root / "controller")
        state.setdefault("wandb_runs", {})["pipeline-controller"] = "pipeline-controller"
        log_pipeline_snapshot(controller, output_root, state)
        save_state(state_path, state)

        if state["phase"] == "screening" and budget.can_start():
            run_screening(base, output_root, group, budget, manifest, state)
            log_pipeline_snapshot(controller, output_root, state)
        if state["phase"] == "teachers" and budget.can_start():
            run_teachers(base, output_root, group, budget, manifest, state)
            log_pipeline_snapshot(controller, output_root, state)
        if state["phase"] == "oof" and budget.can_start(15):
            run_oof(output_root, state)
            log_pipeline_snapshot(controller, output_root, state)
        if state["phase"] == "students" and budget.can_start():
            run_students(base, output_root, group, budget, manifest, state)
            log_pipeline_snapshot(controller, output_root, state)
        if state["phase"] == "selection" and budget.can_start(20):
            run_selection_and_evaluation(output_root, state)
            log_pipeline_snapshot(controller, output_root, state)
        if state["phase"] == "complete":
            controller.log_artifact(
                name="v1-5-pipeline-report",
                files=[
                    state_path,
                    output_root / "data_sources.json",
                    output_root / "merged_train/data_manifest.csv",
                    output_root / "folds.json",
                    output_root / "oof/oof_coverage.json",
                    output_root / "selection/locked_decision.json",
                    output_root / "final/evaluation_complete.json",
                    output_root / "release/release_manifest.json",
                ],
                artifact_type="report",
                aliases=["latest"],
            )
        controller.summary(
            {"phase": state["phase"], "session": state["session"], "completed_tasks": len(state["completed"])}
        )
        controller.finish()
        save_state(state_path, state)
    finally:
        save_state(state_path, state)
        package_state(output_root)
    print(json.dumps({"phase": state["phase"], "session": state["session"]}, indent=2))


if __name__ == "__main__":
    main()
