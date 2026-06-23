import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"
WORKING_ROOT = Path("/kaggle/working")
REPOSITORY_ROOT = WORKING_ROOT / "medical-image-segmentation"
PREPARED_ROOT = WORKING_ROOT / "prepared_data"
RESULTS_ROOT = WORKING_ROOT / "repeated_experiments"
INTERNAL_DATASET_REF = "moon1570/isic-2017-train-val-test-images-and-masks"
EXTERNAL_DATASET_REF = "tntiphan/isic-2018-task-1"
SEEDS = [42, 123, 2026]


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
    import yaml

    config = {
        "experiment_name": "kaggle_repeated_unetplusplus_effb3",
        "seed": SEEDS[0],
        "reproducibility": {"deterministic": True},
        "device": "auto",
        "require_gpu": True,
        "mixed_precision": True,
        "data": {
            "train_images_dir": str(PREPARED_ROOT / "internal/train/images"),
            "train_masks_dir": str(PREPARED_ROOT / "internal/train/masks"),
            "val_images_dir": str(PREPARED_ROOT / "internal/val/images"),
            "val_masks_dir": str(PREPARED_ROOT / "internal/val/masks"),
            "test_images_dir": str(PREPARED_ROOT / "internal/test/images"),
            "test_masks_dir": str(PREPARED_ROOT / "internal/test/masks"),
            "image_size": 384,
        },
        "model": {
            "model_name": "unet_plus_plus",
            "in_channels": 3,
            "out_channels": 1,
            "encoder_name": "efficientnet-b3",
            "encoder_weights": "imagenet",
        },
        "training": {
            "batch_size": 8,
            "epochs": 50,
            "lr": 1.0e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "loss_name": "bce_dice",
            "num_workers": 2,
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "monitor": "val_dice",
                "mode": "max",
            },
        },
        "augmentation": {
            "enabled": True,
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotate": True,
            "color_jitter": True,
        },
        "paths": {
            "output_dir": str(RESULTS_ROOT / "preflight"),
            "checkpoint_dir": str(RESULTS_ROOT / "preflight_checkpoints"),
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


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

    config_path = WORKING_ROOT / "kaggle_repeated_runtime.yaml"
    write_runtime_config(config_path)
    run([sys.executable, "scripts/check_dataset.py", "--config", config_path], cwd=REPOSITORY_ROOT)
    run(
        [
            sys.executable,
            "scripts/run_repeated_experiments.py",
            "--config",
            config_path,
            "--seeds",
            *[str(seed) for seed in SEEDS],
            "--output-root",
            RESULTS_ROOT,
            "--test-images-dir",
            PREPARED_ROOT / "internal/test/images",
            "--test-masks-dir",
            PREPARED_ROOT / "internal/test/masks",
            "--external-images-dir",
            PREPARED_ROOT / "external/images",
            "--external-masks-dir",
            PREPARED_ROOT / "external/masks",
        ],
        cwd=REPOSITORY_ROOT,
    )

    manifest_path = RESULTS_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best_seed = manifest["best_seed_by_validation_dice"]
    checkpoint = RESULTS_ROOT / f"seed_{best_seed}/checkpoints/best_model.pth"
    best_config = RESULTS_ROOT / f"runtime_configs/seed_{best_seed}.yaml"
    run(
        [
            sys.executable,
            "scripts/benchmark_model.py",
            "--config",
            best_config,
            "--checkpoint",
            checkpoint,
            "--devices",
            "cpu",
            "cuda",
            "--warmup",
            "5",
            "--cpu-iterations",
            "10",
            "--cuda-iterations",
            "50",
            "--output-dir",
            RESULTS_ROOT / "benchmark",
        ],
        cwd=REPOSITORY_ROOT,
    )

    execution_manifest = {
        "repository_commit": commit,
        "internal_dataset": INTERNAL_DATASET_REF,
        "external_dataset": EXTERNAL_DATASET_REF,
        "seeds": SEEDS,
        "best_seed_by_validation_dice": best_seed,
        "training_device_required": "cuda",
        "benchmark_devices": ["cpu", "cuda"],
    }
    (RESULTS_ROOT / "execution_manifest.json").write_text(
        json.dumps(execution_manifest, indent=2), encoding="utf-8"
    )
    shutil.copy2(PREPARED_ROOT / "internal/preparation_report.json", RESULTS_ROOT / "internal_data_report.json")
    shutil.copy2(PREPARED_ROOT / "external/preparation_report.json", RESULTS_ROOT / "external_data_report.json")

    for last_checkpoint in RESULTS_ROOT.glob("seed_*/checkpoints/last_model.pth"):
        last_checkpoint.unlink()
    shutil.rmtree(PREPARED_ROOT)
    print("Repeated experiment and benchmark completed:", RESULTS_ROOT, flush=True)


if __name__ == "__main__":
    main()
