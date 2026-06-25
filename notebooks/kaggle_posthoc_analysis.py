import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = WORKING_ROOT / "medical-image-segmentation"
PREPARED_ROOT = WORKING_ROOT / "prepared_data"
RESULTS_ROOT = WORKING_ROOT / "posthoc_analysis"
CHECKPOINT_PATH = WORKING_ROOT / "best_model.pth"
CHECKPOINT_URL = "https://github.com/dongtingshuo/medical-image-segmentation/releases/download/v1.0.0/best_model.pth"
INTERNAL_DATASET_REF = "moon1570/isic-2017-train-val-test-images-and-masks"
EXTERNAL_DATASET_REF = "tntiphan/isic-2018-task-1"


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


def write_runtime_config(path):
    config = {
        "experiment_name": "posthoc_threshold_failure_analysis",
        "seed": 42,
        "reproducibility": {"deterministic": True},
        "device": "auto",
        "require_gpu": False,
        "mixed_precision": False,
        "data": {
            "train_images_dir": str(PREPARED_ROOT / "internal/train/images"),
            "train_masks_dir": str(PREPARED_ROOT / "internal/train/masks"),
            "val_images_dir": str(PREPARED_ROOT / "internal/val/images"),
            "val_masks_dir": str(PREPARED_ROOT / "internal/val/masks"),
            "test_images_dir": str(PREPARED_ROOT / "internal/test/images"),
            "test_masks_dir": str(PREPARED_ROOT / "internal/test/masks"),
            "external_images_dir": str(PREPARED_ROOT / "external/images"),
            "external_masks_dir": str(PREPARED_ROOT / "external/masks"),
            "image_size": 384,
        },
        "model": {
            "model_name": "unet_plus_plus",
            "in_channels": 3,
            "out_channels": 1,
            "encoder_name": "efficientnet-b3",
            "encoder_weights": None,
        },
        "training": {
            "batch_size": 8,
            "epochs": 1,
            "lr": 1.0e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "loss_name": "bce_dice",
            "num_workers": 2,
        },
        "augmentation": {"enabled": False},
        "paths": {
            "output_dir": str(RESULTS_ROOT),
            "checkpoint_dir": str(RESULTS_ROOT / "checkpoints"),
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def read_best_threshold(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No threshold rows found in {path}")
    best = max(rows, key=lambda row: (float(row["dice"]), float(row["threshold"])))
    return float(best["threshold"]), best


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/kaggle/working/matplotlib-cache")
    internal_dataset = resolve_dataset(INTERNAL_DATASET_REF, "ISIC 2017 internal dataset")
    external_dataset = resolve_dataset(EXTERNAL_DATASET_REF, "ISIC 2018 external dataset")

    if REPOSITORY_ROOT.exists():
        shutil.rmtree(REPOSITORY_ROOT)
    run(["git", "clone", "--depth", "1", REPOSITORY_URL, REPOSITORY_ROOT])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    print("Repository commit:", commit, flush=True)

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], cwd=REPOSITORY_ROOT)
    run([sys.executable, "scripts/kaggle_prepare_gpu.py", "--install-if-needed"], cwd=REPOSITORY_ROOT)
    run([sys.executable, "-m", "pytest", "-q"], cwd=REPOSITORY_ROOT)
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
            "384",
        ],
        cwd=REPOSITORY_ROOT,
    )

    print(f"Downloading checkpoint from {CHECKPOINT_URL}", flush=True)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    config_path = WORKING_ROOT / "posthoc_runtime.yaml"
    write_runtime_config(config_path)

    threshold_dir = RESULTS_ROOT / "threshold_search"
    run(
        [
            sys.executable,
            "scripts/search_threshold.py",
            "--config",
            config_path,
            "--checkpoint",
            CHECKPOINT_PATH,
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
    best_threshold, best_row = read_best_threshold(threshold_dir / "threshold_search.csv")
    print("Best validation threshold:", best_threshold, best_row, flush=True)

    for split in ["test", "external"]:
        run(
            [
                sys.executable,
                "scripts/analyze_failure_cases.py",
                "--config",
                config_path,
                "--checkpoint",
                CHECKPOINT_PATH,
                "--split",
                split,
                "--threshold",
                f"{best_threshold:.6f}",
                "--top-k",
                "16",
                "--sort-by",
                "dice",
                "--output-dir",
                RESULTS_ROOT / f"failure_cases_{split}",
            ],
            cwd=REPOSITORY_ROOT,
        )

    manifest = {
        "repository_commit": commit,
        "checkpoint_url": CHECKPOINT_URL,
        "internal_dataset": INTERNAL_DATASET_REF,
        "external_dataset": EXTERNAL_DATASET_REF,
        "best_threshold": best_threshold,
        "best_threshold_row": best_row,
        "outputs": {
            "threshold_search": str(threshold_dir),
            "failure_cases_test": str(RESULTS_ROOT / "failure_cases_test"),
            "failure_cases_external": str(RESULTS_ROOT / "failure_cases_external"),
        },
    }
    (RESULTS_ROOT / "posthoc_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(config_path, RESULTS_ROOT / "posthoc_runtime.yaml")
    shutil.rmtree(PREPARED_ROOT)
    CHECKPOINT_PATH.unlink(missing_ok=True)
    print("Post-hoc threshold search and failure analysis completed:", RESULTS_ROOT, flush=True)


if __name__ == "__main__":
    main()
