import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = WORKING_ROOT / "medical-image-segmentation"
PREPARED_ROOT = WORKING_ROOT / "prepared_data"
RESULTS_ROOT = WORKING_ROOT / "research_v1_4_aggressive"
INTERNAL_DATASET_REF = "moon1570/isic-2017-train-val-test-images-and-masks"
EXTERNAL_DATASET_REF = "tntiphan/isic-2018-task-1"
FULL_CONFIG = "configs/kaggle_aggressive_v1_4.yaml"
DEBUG_CONFIG = "configs/kaggle_aggressive_v1_4_debug.yaml"
SPLITS = ["val", "test", "external"]

FULL_EXPERIMENTS = [
    {
        "name": "unetpp_effb4_448",
        "stage": 1,
        "model_name": "unet_plus_plus",
        "encoder_name": "efficientnet-b4",
        "image_size": 448,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "epochs": 24,
        "lr": 8.0e-5,
    },
    {
        "name": "unetpp_effb4_512_finetune",
        "stage": 2,
        "model_name": "unet_plus_plus",
        "encoder_name": "efficientnet-b4",
        "image_size": 512,
        "batch_size": 3,
        "gradient_accumulation_steps": 3,
        "epochs": 12,
        "lr": 3.0e-5,
        "resume_from_experiment": "unetpp_effb4_448",
        "resume_optimizer": False,
        "resume_training_state": False,
    },
    {
        "name": "deeplabv3plus_effb4_448",
        "stage": 3,
        "model_name": "deeplabv3plus",
        "encoder_name": "efficientnet-b4",
        "image_size": 448,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "epochs": 22,
        "lr": 8.0e-5,
    },
    {
        "name": "unetpp_effb5_448",
        "stage": 4,
        "model_name": "unet_plus_plus",
        "encoder_name": "efficientnet-b5",
        "image_size": 448,
        "batch_size": 3,
        "gradient_accumulation_steps": 3,
        "epochs": 20,
        "lr": 6.0e-5,
    },
]

DEBUG_EXPERIMENTS = [
    {
        "name": "debug_unetpp_effb3_256",
        "stage": 1,
        "model_name": "unet_plus_plus",
        "encoder_name": "efficientnet-b3",
        "image_size": 256,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "epochs": 2,
        "lr": 1.0e-4,
    }
]

TTA_VARIANTS = [
    {"name": "fast", "scales": "1.0", "horizontal_flip": False, "vertical_flip": False, "min_component_area": 0, "fill_holes": False},
    {"name": "flip_tta", "scales": "1.0", "horizontal_flip": True, "vertical_flip": True, "min_component_area": 0, "fill_holes": False},
    {"name": "best_accuracy", "scales": "0.875,1.0,1.125", "horizontal_flip": True, "vertical_flip": True, "min_component_area": 64, "fill_holes": True},
]


def run(command, cwd=None):
    print(">>>", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def resolve_dataset(dataset_ref, label):
    owner, slug = dataset_ref.split("/", maxsplit=1)
    candidates = [
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/datasets") / owner / slug,
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"{label}: {candidate}", flush=True)
            return candidate
    available = sorted(str(path) for path in Path("/kaggle/input").glob("*/*/*"))
    raise FileNotFoundError(f"{label} ({dataset_ref}) is not mounted. Detected inputs: {available}")


def is_oom_log(text):
    lowered = text.lower()
    return "out of memory" in lowered or "cuda error: out of memory" in lowered


def experiment_config(base_config, experiment, output_root):
    config = json.loads(json.dumps(base_config))
    config["experiment_name"] = f"kaggle_aggressive_v1_4_{experiment['name']}"
    config.setdefault("data", {})["image_size"] = int(experiment["image_size"])
    config.setdefault("model", {})
    config["model"]["model_name"] = experiment["model_name"]
    config["model"]["encoder_name"] = experiment["encoder_name"]
    config["model"]["encoder_weights"] = "imagenet"
    config.setdefault("training", {})
    config["training"]["batch_size"] = int(experiment["batch_size"])
    config["training"]["gradient_accumulation_steps"] = int(experiment.get("gradient_accumulation_steps", 1))
    config["training"]["epochs"] = int(experiment["epochs"])
    config["training"]["lr"] = float(experiment["lr"])
    config["training"]["loss_name"] = "bce_dice"
    config["training"]["resume_optimizer"] = bool(experiment.get("resume_optimizer", True))
    config["training"]["resume_training_state"] = bool(experiment.get("resume_training_state", True))
    config["loss"] = {"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5}
    config.setdefault("augmentation", {}).setdefault("low_contrast", {})["enabled"] = True
    config.setdefault("paths", {})
    config["paths"]["output_dir"] = str(output_root / experiment["name"] / "outputs")
    config["paths"]["checkpoint_dir"] = str(output_root / experiment["name"] / "checkpoints")
    return config


def train_command(config_path, resume_path=None):
    command = [sys.executable, "train.py", "--config", str(config_path)]
    if resume_path is not None and Path(resume_path).exists():
        command.extend(["--resume", str(resume_path)])
    return command


def train_with_oom_fallback(config_path, config, log_path, resume_path=None):
    command = train_command(config_path, resume_path=resume_path)
    print(">>>", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
    if result.returncode == 0:
        print(result.stdout, flush=True)
        return config

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    batch_size = int(config.get("training", {}).get("batch_size", 1))
    if batch_size > 1 and is_oom_log(combined):
        new_batch_size = max(1, batch_size // 2)
        accumulation = int(config.get("training", {}).get("gradient_accumulation_steps", 1))
        config["training"]["batch_size"] = new_batch_size
        config["training"]["gradient_accumulation_steps"] = max(accumulation, int(round(batch_size / new_batch_size)))
        write_yaml(config_path, config)
        print(
            f"CUDA OOM detected. Retrying with batch_size={new_batch_size}, "
            f"gradient_accumulation_steps={config['training']['gradient_accumulation_steps']}.",
            flush=True,
        )
        retry_resume = resume_path if resume_path is not None and Path(resume_path).exists() else None
        retry_command = train_command(config_path, resume_path=retry_resume)
        print(">>>", " ".join(retry_command), flush=True)
        result = subprocess.run(retry_command, cwd=REPOSITORY_ROOT, text=True, capture_output=True)
        log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
        if result.returncode == 0:
            print(result.stdout, flush=True)
            return config

    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    raise subprocess.CalledProcessError(result.returncode, command)


def read_best_threshold(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No threshold rows found in {path}")
    best = max(rows, key=lambda row: (float(row["dice"]), float(row["threshold"])))
    return float(best["threshold"])


def run_threshold_search(config_path, checkpoint_path, output_dir):
    run(
        [
            sys.executable,
            "scripts/search_threshold.py",
            "--config",
            config_path,
            "--checkpoint",
            checkpoint_path,
            "--split",
            "val",
            "--start",
            "0.25",
            "--stop",
            "0.75",
            "--step",
            "0.025",
            "--output-dir",
            output_dir,
        ],
        cwd=REPOSITORY_ROOT,
    )
    return read_best_threshold(Path(output_dir) / "threshold_search.csv")


def evaluate_split(config_path, checkpoint_path, split, output_dir, threshold):
    config = load_yaml(config_path)
    config.setdefault("paths", {})["output_dir"] = str(output_dir)
    eval_config_path = output_dir / f"eval_{split}_config.yaml"
    write_yaml(eval_config_path, config)
    run(
        [
            sys.executable,
            "evaluate.py",
            "--config",
            eval_config_path,
            "--checkpoint",
            checkpoint_path,
            "--split",
            split,
            "--threshold",
            f"{threshold:.6f}",
        ],
        cwd=REPOSITORY_ROOT,
    )


def evaluate_tta_variants(config_path, checkpoint_path, experiment_root, threshold, debug=False):
    tta_variants = TTA_VARIANTS[:1] if debug else TTA_VARIANTS
    for variant in tta_variants:
        for split in ["test", "external"]:
            output_dir = experiment_root / f"tta_{variant['name']}_{split}"
            command = [
                sys.executable,
                "scripts/evaluate_tta_postprocess.py",
                "--config",
                config_path,
                "--checkpoint",
                checkpoint_path,
                "--split",
                split,
                "--threshold",
                f"{threshold:.6f}",
                "--scales",
                variant["scales"],
                "--min-component-area",
                str(variant["min_component_area"]),
                "--output-dir",
                output_dir,
            ]
            if variant["horizontal_flip"]:
                command.append("--horizontal-flip")
            if variant["vertical_flip"]:
                command.append("--vertical-flip")
            if variant["fill_holes"]:
                command.append("--fill-holes")
            run(command, cwd=REPOSITORY_ROOT)


def mark_failed(path, experiment, exc):
    payload = {
        "experiment": experiment["name"],
        "stage": experiment.get("stage"),
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(experiment, base_config, experiments_root, debug=False, retry_failed=False):
    experiment_root = experiments_root / experiment["name"]
    completed_path = experiment_root / "completed.json"
    failed_path = experiment_root / "failed.json"
    if completed_path.exists():
        print(f"Experiment `{experiment['name']}` already completed. Skipping.", flush=True)
        return json.loads(completed_path.read_text(encoding="utf-8"))
    if failed_path.exists() and not retry_failed:
        print(f"Experiment `{experiment['name']}` has failed.json. Skipping; pass --retry-failed to retry.", flush=True)
        return json.loads(failed_path.read_text(encoding="utf-8"))
    if failed_path.exists() and retry_failed:
        failed_path.unlink()

    config_path = experiment_root / "runtime_config.yaml"
    if config_path.exists():
        config = load_yaml(config_path)
    else:
        config = experiment_config(base_config, experiment, experiments_root)
        write_yaml(config_path, config)

    resume_path = experiment_root / "checkpoints/last_model.pth"
    if not resume_path.exists() and experiment.get("resume_from_experiment"):
        parent_best = experiments_root / experiment["resume_from_experiment"] / "checkpoints/best_model.pth"
        if not parent_best.exists():
            raise FileNotFoundError(
                f"Fine-tune source checkpoint is missing for {experiment['name']}: {parent_best}"
            )
        resume_path = parent_best
    trained_config = train_with_oom_fallback(
        config_path,
        config,
        experiment_root / "train.log",
        resume_path=resume_path if resume_path.exists() else None,
    )
    write_yaml(config_path, trained_config)
    checkpoint_path = experiment_root / "checkpoints/best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = experiment_root / "checkpoints/last_model.pth"
    threshold = run_threshold_search(config_path, checkpoint_path, experiment_root / "threshold_search")
    trained_config.setdefault("inference", {})["threshold"] = threshold
    write_yaml(config_path, trained_config)
    for split in SPLITS:
        evaluate_split(config_path, checkpoint_path, split, experiment_root / f"evaluation_{split}", threshold)
    evaluate_tta_variants(config_path, checkpoint_path, experiment_root, threshold, debug=debug)
    summary = {
        "experiment": experiment["name"],
        "stage": experiment.get("stage"),
        "checkpoint": str(checkpoint_path),
        "best_threshold": threshold,
        "batch_size": trained_config["training"]["batch_size"],
        "gradient_accumulation_steps": trained_config["training"].get("gradient_accumulation_steps", 1),
    }
    completed_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def collect_metric_row(path, extra):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return {**extra, **rows[0]}


def write_summary(experiments_root, output_dir):
    rows = []
    for completed_path in sorted(experiments_root.glob("*/completed.json")):
        experiment = completed_path.parent.name
        summary = json.loads(completed_path.read_text(encoding="utf-8"))
        for split in SPLITS:
            rows.append(
                collect_metric_row(
                    completed_path.parent / f"evaluation_{split}/metrics.csv",
                    {"experiment": experiment, "stage": summary.get("stage", ""), "mode": "single_model"},
                )
            )
        for tta_path in sorted(completed_path.parent.glob("tta_*_*/tta_postprocess_metrics.csv")):
            name = tta_path.parent.name.replace("tta_", "")
            parts = name.rsplit("_", maxsplit=1)
            rows.append(
                collect_metric_row(
                    tta_path,
                    {
                        "experiment": experiment,
                        "stage": summary.get("stage", ""),
                        "mode": parts[0] if len(parts) == 2 else name,
                    },
                )
            )
        for ensemble_path in sorted(completed_path.parent.glob("ensemble_*/ensemble_metrics.csv")):
            rows.append(
                collect_metric_row(
                    ensemble_path,
                    {
                        "experiment": experiment,
                        "stage": summary.get("stage", ""),
                        "mode": "ensemble",
                    },
                )
            )
    rows = [row for row in rows if row]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "aggressive_v1_4_summary.csv"
    json_path = output_dir / "aggressive_v1_4_summary.json"
    md_path = output_dir / "aggressive_v1_4_summary.md"
    if not rows:
        csv_path.write_text("experiment,mode,split,dice,iou,precision,recall,boundary_f1\n", encoding="utf-8")
        json_path.write_text("[]\n", encoding="utf-8")
        md_path.write_text("# Aggressive v1.4 Summary\n\nNo completed metric rows were found.\n", encoding="utf-8")
        return {"csv": csv_path, "json": json_path, "markdown": md_path}
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Aggressive v1.4 Summary",
        "",
        "| Experiment | Mode | Split | Dice | IoU | Precision | Recall | Boundary F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item.get("split", ""), item.get("experiment", ""), item.get("mode", ""))):
        lines.append(
            "| {experiment} | {mode} | {split} | {dice:.6f} | {iou:.6f} | {precision:.6f} | "
            "{recall:.6f} | {boundary_f1:.6f} |".format(
                experiment=row.get("experiment", ""),
                mode=row.get("mode", ""),
                split=row.get("split", ""),
                dice=float(row.get("dice", 0.0)),
                iou=float(row.get("iou", 0.0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                boundary_f1=float(row.get("boundary_f1", 0.0)),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def run_pair_ensemble(experiments_root):
    member_names = ["unetpp_effb4_448", "deeplabv3plus_effb4_448"]
    member_args = []
    for name in member_names:
        root = experiments_root / name
        if not (root / "completed.json").exists():
            print(f"Skipping ensemble because `{name}` is not completed.", flush=True)
            return
        checkpoint = root / "checkpoints/best_model.pth"
        if not checkpoint.exists():
            checkpoint = root / "checkpoints/last_model.pth"
        member_args.extend(["--member", f"{root / 'runtime_config.yaml'}:{checkpoint}"])
    ensemble_root = experiments_root / "unetpp_effb4_448"
    threshold = 0.5
    threshold_csv = ensemble_root / "threshold_search/threshold_search.csv"
    if threshold_csv.exists():
        threshold = read_best_threshold(threshold_csv)
    for split in ["test", "external"]:
        run(
            [
                sys.executable,
                "scripts/evaluate_ensemble.py",
                *member_args,
                "--split",
                split,
                "--threshold",
                f"{threshold:.6f}",
                "--output-dir",
                ensemble_root / f"ensemble_b4_unetpp_deeplab_{split}",
            ],
            cwd=REPOSITORY_ROOT,
        )


def main():
    parser = argparse.ArgumentParser(description="Run aggressive v1.4 Kaggle segmentation experiments.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/kaggle/working/matplotlib-cache")
    internal_dataset = resolve_dataset(INTERNAL_DATASET_REF, "ISIC 2017 internal dataset")
    external_dataset = resolve_dataset(EXTERNAL_DATASET_REF, "ISIC 2018 external dataset")

    if REPOSITORY_ROOT.exists():
        shutil.rmtree(REPOSITORY_ROOT)
    run(["git", "clone", "--depth", "1", REPOSITORY_URL, REPOSITORY_ROOT])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    print("Repository commit:", commit, flush=True)

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], cwd=REPOSITORY_ROOT)
    run([sys.executable, "scripts/kaggle_prepare_gpu.py", "--install-if-needed"], cwd=REPOSITORY_ROOT)
    run([sys.executable, "-m", "pytest", "-q"], cwd=REPOSITORY_ROOT)

    config_name = DEBUG_CONFIG if args.debug else FULL_CONFIG
    base_config = load_yaml(REPOSITORY_ROOT / config_name)
    experiments = DEBUG_EXPERIMENTS if args.debug else FULL_EXPERIMENTS
    prepare_image_size = max(int(item["image_size"]) for item in experiments)
    run(
        [
            sys.executable,
            "scripts/prepare_isic_data.py",
            "--internal-root",
            internal_dataset,
            "--external-root",
            external_dataset,
            "--output-root",
            PREPARED_ROOT,
            "--image-size",
            str(prepare_image_size),
        ],
        cwd=REPOSITORY_ROOT,
    )
    check_config_path = write_yaml(WORKING_ROOT / "kaggle_aggressive_v1_4_check.yaml", base_config)
    run([sys.executable, "scripts/check_dataset.py", "--config", check_config_path], cwd=REPOSITORY_ROOT)

    experiments_root = RESULTS_ROOT / "experiments"
    summaries = []
    for experiment in sorted(experiments, key=lambda item: item.get("stage", 0)):
        try:
            summaries.append(
                run_experiment(
                    experiment,
                    base_config,
                    experiments_root,
                    debug=args.debug,
                    retry_failed=args.retry_failed,
                )
            )
        except Exception as exc:  # noqa: BLE001
            mark_failed(experiments_root / experiment["name"] / "failed.json", experiment, exc)
            print(f"Experiment `{experiment['name']}` failed: {exc}", flush=True)

    if not args.debug:
        run_pair_ensemble(experiments_root)
    summary_outputs = write_summary(experiments_root, RESULTS_ROOT / "comparison")
    execution_manifest = {
        "repository_commit": commit,
        "internal_dataset": INTERNAL_DATASET_REF,
        "external_dataset": EXTERNAL_DATASET_REF,
        "debug": args.debug,
        "experiments": experiments,
        "experiment_summaries": summaries,
        "splits": SPLITS,
        "summary_outputs": {key: str(value) for key, value in summary_outputs.items()},
        "training_device_required": "cuda",
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "execution_manifest.json").write_text(json.dumps(execution_manifest, indent=2), encoding="utf-8")
    shutil.copy2(check_config_path, RESULTS_ROOT / "kaggle_aggressive_v1_4_runtime_base.yaml")
    run(
        [
            sys.executable,
            "scripts/package_release_artifacts.py",
            "--source-root",
            WORKING_ROOT,
            "--output-dir",
            WORKING_ROOT / "release_artifacts",
            "--package-name",
            "medical-segmentation-aggressive-artifacts-v1.4",
        ],
        cwd=REPOSITORY_ROOT,
    )
    shutil.rmtree(PREPARED_ROOT)
    print("Aggressive v1.4 workflow completed:", RESULTS_ROOT, flush=True)


if __name__ == "__main__":
    main()
