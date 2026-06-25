import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = WORKING_ROOT / "medical-image-segmentation"
PREPARED_ROOT = WORKING_ROOT / "prepared_data"
RESULTS_ROOT = WORKING_ROOT / "research_v1_3_low_contrast"
INTERNAL_DATASET_REF = "moon1570/isic-2017-train-val-test-images-and-masks"
EXTERNAL_DATASET_REF = "tntiphan/isic-2018-task-1"
FULL_CONFIG = "configs/kaggle_low_contrast_v1_3.yaml"
DEBUG_CONFIG = "configs/kaggle_low_contrast_v1_3_debug.yaml"
FULL_VARIANTS = ["control_bce_dice", "contrast_aug_bce_dice", "contrast_aug_focal_dice", "contrast_aug_tversky"]
DEBUG_VARIANTS = ["control_bce_dice", "contrast_aug_bce_dice"]
SPLITS = ["val", "test", "external"]


def run(command, cwd=None):
    print(">>>", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


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
    raise FileNotFoundError(
        f"{label} ({dataset_ref}) is not mounted at any supported path. Detected inputs: {available}"
    )


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def variant_config(base_config, variant, output_root):
    config = json.loads(json.dumps(base_config))
    config["experiment_name"] = f"kaggle_low_contrast_v1_3_{variant}"
    config.setdefault("paths", {})
    config["paths"]["output_dir"] = str(output_root / variant / "outputs")
    config["paths"]["checkpoint_dir"] = str(output_root / variant / "checkpoints")
    config.setdefault("augmentation", {}).setdefault("low_contrast", {})
    config.setdefault("loss", {})
    config.setdefault("training", {})

    low_contrast_enabled = variant.startswith("contrast_aug")
    config["augmentation"]["low_contrast"]["enabled"] = low_contrast_enabled
    if variant == "contrast_aug_focal_dice":
        config["training"]["loss_name"] = "focal_dice"
        config["loss"] = {"name": "focal_dice", "alpha": 0.25, "gamma": 2.0}
    elif variant == "contrast_aug_tversky":
        config["training"]["loss_name"] = "tversky"
        config["loss"] = {"name": "tversky", "alpha": 0.3, "beta": 0.7, "smooth": 1.0}
    else:
        config["training"]["loss_name"] = "bce_dice"
        config["loss"] = {"name": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5}
    return config


def read_best_threshold(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No threshold rows found in {path}")
    best = max(rows, key=lambda row: (float(row["dice"]), float(row["threshold"])))
    return float(best["threshold"])


def is_oom_log(text):
    lowered = text.lower()
    return "out of memory" in lowered or "cuda error: out of memory" in lowered


def train_with_oom_fallback(config_path, config, cwd, log_path):
    command = [sys.executable, "train.py", "--config", str(config_path)]
    print(">>>", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
    if result.returncode == 0:
        print(result.stdout, flush=True)
        return config
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    batch_size = int(config.get("training", {}).get("batch_size", 8))
    if batch_size > 4 and is_oom_log(combined):
        print("CUDA OOM detected. Retrying with batch_size=4.", flush=True)
        config["training"]["batch_size"] = 4
        write_yaml(config_path, config)
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
        if result.returncode == 0:
            print(result.stdout, flush=True)
            return config
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    raise subprocess.CalledProcessError(result.returncode, command)


def move_training_metrics(variant_root):
    metrics_path = variant_root / "outputs/metrics.csv"
    if metrics_path.exists():
        metrics_path.replace(variant_root / "outputs/training_history.csv")


def evaluate_variant(config_path, checkpoint_path, split, output_dir):
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
            str(config.get("inference", {}).get("threshold", 0.5)),
        ],
        cwd=REPOSITORY_ROOT,
    )


def run_variant(variant, base_config, output_root):
    variant_root = output_root / variant
    if variant_root.exists():
        shutil.rmtree(variant_root)
    config = variant_config(base_config, variant, output_root)
    config_path = write_yaml(variant_root / "runtime_config.yaml", config)
    trained_config = train_with_oom_fallback(config_path, config, REPOSITORY_ROOT, variant_root / "train.log")
    write_yaml(config_path, trained_config)
    move_training_metrics(variant_root)
    checkpoint_path = variant_root / "checkpoints/best_model.pth"

    threshold_dir = variant_root / "threshold_search"
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
            "0.30",
            "--stop",
            "0.70",
            "--step",
            "0.05",
            "--output-dir",
            threshold_dir,
        ],
        cwd=REPOSITORY_ROOT,
    )
    best_threshold = read_best_threshold(threshold_dir / "threshold_search.csv")
    trained_config.setdefault("inference", {})["threshold"] = best_threshold
    write_yaml(config_path, trained_config)

    for split in SPLITS:
        evaluate_variant(config_path, checkpoint_path, split, variant_root / f"evaluation_{split}")
        run(
            [
                sys.executable,
                "scripts/analyze_subgroups.py",
                "--config",
                config_path,
                "--checkpoint",
                checkpoint_path,
                "--split",
                split,
                "--threshold",
                f"{best_threshold:.6f}",
                "--output-dir",
                variant_root / f"subgroup_{split}",
            ],
            cwd=REPOSITORY_ROOT,
        )
    return {"variant": variant, "best_threshold": best_threshold, "batch_size": trained_config["training"]["batch_size"]}


def main():
    parser = argparse.ArgumentParser(description="Run v1.3 low-contrast specialist experiments on Kaggle.")
    parser.add_argument("--debug", action="store_true")
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
            str(base_config["data"]["image_size"]),
        ],
        cwd=REPOSITORY_ROOT,
    )
    check_config_path = write_yaml(WORKING_ROOT / "kaggle_low_contrast_v1_3_check.yaml", base_config)
    run([sys.executable, "scripts/check_dataset.py", "--config", check_config_path], cwd=REPOSITORY_ROOT)

    variants = DEBUG_VARIANTS if args.debug else FULL_VARIANTS
    variant_summaries = [run_variant(variant, base_config, RESULTS_ROOT / "variants") for variant in variants]
    run(
        [
            sys.executable,
            "scripts/compare_low_contrast_variants.py",
            "--variants-root",
            RESULTS_ROOT / "variants",
            "--variants",
            *variants,
            "--splits",
            *SPLITS,
            "--baseline-variant",
            "control_bce_dice",
            "--target-split",
            "test",
            "--output-dir",
            RESULTS_ROOT / "comparison",
        ],
        cwd=REPOSITORY_ROOT,
    )

    execution_manifest = {
        "repository_commit": commit,
        "internal_dataset": INTERNAL_DATASET_REF,
        "external_dataset": EXTERNAL_DATASET_REF,
        "debug": args.debug,
        "variants": variants,
        "variant_summaries": variant_summaries,
        "splits": SPLITS,
        "image_size": base_config["data"]["image_size"],
        "training_device_required": "cuda",
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "execution_manifest.json").write_text(json.dumps(execution_manifest, indent=2), encoding="utf-8")
    shutil.copy2(check_config_path, RESULTS_ROOT / "kaggle_low_contrast_v1_3_runtime_base.yaml")
    run(
        [
            sys.executable,
            "scripts/package_release_artifacts.py",
            "--source-root",
            WORKING_ROOT,
            "--output-dir",
            WORKING_ROOT / "release_artifacts",
            "--package-name",
            "medical-segmentation-low-contrast-artifacts-v1.3",
        ],
        cwd=REPOSITORY_ROOT,
    )
    shutil.rmtree(PREPARED_ROOT)
    print("Low-contrast v1.3 workflow completed:", RESULTS_ROOT, flush=True)


if __name__ == "__main__":
    main()
