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
RESULTS_ROOT = WORKING_ROOT / "research_v1_2"
INTERNAL_DATASET_REF = "moon1570/isic-2017-train-val-test-images-and-masks"
EXTERNAL_DATASET_REF = "tntiphan/isic-2018-task-1"

CV_FOLDS = 3
CV_EPOCHS = 15
CV_PATIENCE = 4
ENCODER_EPOCHS = 20
ENCODER_PATIENCE = 5
ENCODERS = ["efficientnet-b3", "resnet34"]


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


def copy_split_into_trainval(split, target_images, target_masks):
    source_images = PREPARED_ROOT / f"internal/{split}/images"
    source_masks = PREPARED_ROOT / f"internal/{split}/masks"
    mask_map = {path.stem: path for path in source_masks.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    for image_path in sorted(source_images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_map.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for trainval copy: {image_path.stem} in {source_masks}")
        new_name = f"{split}_{image_path.stem}"
        shutil.copy2(image_path, target_images / f"{new_name}{image_path.suffix.lower()}")
        shutil.copy2(mask_path, target_masks / f"{new_name}{mask_path.suffix.lower()}")


def make_trainval_dirs():
    target_images = PREPARED_ROOT / "internal/trainval/images"
    target_masks = PREPARED_ROOT / "internal/trainval/masks"
    target_images.mkdir(parents=True, exist_ok=True)
    target_masks.mkdir(parents=True, exist_ok=True)
    copy_split_into_trainval("train", target_images, target_masks)
    copy_split_into_trainval("val", target_images, target_masks)
    return target_images, target_masks


def write_runtime_config(path):
    config = {
        "experiment_name": "kaggle_research_v1_2_unetplusplus_effb3",
        "seed": 42,
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
            "external_images_dir": str(PREPARED_ROOT / "external/images"),
            "external_masks_dir": str(PREPARED_ROOT / "external/masks"),
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
            "epochs": 20,
            "lr": 1.0e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "loss_name": "bce_dice",
            "num_workers": 2,
            "early_stopping": {
                "enabled": True,
                "patience": 5,
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


def best_encoder_artifact(output_root):
    manifest = json.loads((output_root / "encoder_comparison_manifest.json").read_text(encoding="utf-8"))
    best_encoder = manifest["best_encoder_by_validation_dice"]
    key = best_encoder.replace("/", "_")
    return best_encoder, output_root / key / "checkpoints/best_model.pth", output_root / "runtime_configs" / f"{key}.yaml"


def read_best_threshold(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No threshold rows found in {path}")
    best = max(rows, key=lambda row: (float(row["dice"]), float(row["threshold"])))
    return float(best["threshold"])


def main():
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

    config_path = WORKING_ROOT / "kaggle_research_v1_2_runtime.yaml"
    write_runtime_config(config_path)
    run([sys.executable, "scripts/check_dataset.py", "--config", config_path], cwd=REPOSITORY_ROOT)
    trainval_images, trainval_masks = make_trainval_dirs()

    run(
        [
            sys.executable,
            "scripts/run_cross_validation.py",
            "--config",
            config_path,
            "--images-dir",
            trainval_images,
            "--masks-dir",
            trainval_masks,
            "--output-root",
            RESULTS_ROOT / "cross_validation",
            "--k",
            str(CV_FOLDS),
            "--epochs",
            str(CV_EPOCHS),
            "--patience",
            str(CV_PATIENCE),
        ],
        cwd=REPOSITORY_ROOT,
    )
    run(
        [
            sys.executable,
            "scripts/run_encoder_comparison.py",
            "--config",
            config_path,
            "--output-root",
            RESULTS_ROOT / "encoder_comparison",
            "--encoders",
            *ENCODERS,
            "--epochs",
            str(ENCODER_EPOCHS),
            "--patience",
            str(ENCODER_PATIENCE),
        ],
        cwd=REPOSITORY_ROOT,
    )

    best_encoder, checkpoint, best_config = best_encoder_artifact(RESULTS_ROOT / "encoder_comparison")
    threshold_dir = RESULTS_ROOT / "threshold_search"
    run(
        [
            sys.executable,
            "scripts/search_threshold.py",
            "--config",
            best_config,
            "--checkpoint",
            checkpoint,
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
    for split in ["test", "external"]:
        run(
            [
                sys.executable,
                "scripts/analyze_subgroups.py",
                "--config",
                best_config,
                "--checkpoint",
                checkpoint,
                "--split",
                split,
                "--threshold",
                f"{best_threshold:.6f}",
                "--output-dir",
                RESULTS_ROOT / f"subgroup_analysis_{split}",
            ],
            cwd=REPOSITORY_ROOT,
        )

    run(
        [
            sys.executable,
            "scripts/analyze_statistics.py",
            "--inputs",
            RESULTS_ROOT / "cross_validation/cross_validation_metrics.csv",
            RESULTS_ROOT / "encoder_comparison/encoder_comparison_metrics.csv",
            "--output-dir",
            RESULTS_ROOT / "statistics_cv_encoder",
            "--group-by",
            "source",
            "split",
        ],
        cwd=REPOSITORY_ROOT,
    )
    for split in ["test", "external"]:
        run(
            [
                sys.executable,
                "scripts/analyze_statistics.py",
                "--inputs",
                RESULTS_ROOT / f"subgroup_analysis_{split}/subgroup_per_sample.csv",
                "--output-dir",
                RESULTS_ROOT / f"statistics_subgroup_{split}",
                "--group-by",
                "split",
                "lesion_size_group",
            ],
            cwd=REPOSITORY_ROOT,
        )

    execution_manifest = {
        "repository_commit": commit,
        "internal_dataset": INTERNAL_DATASET_REF,
        "external_dataset": EXTERNAL_DATASET_REF,
        "cv_folds": CV_FOLDS,
        "cv_epochs": CV_EPOCHS,
        "encoder_epochs": ENCODER_EPOCHS,
        "encoders": ENCODERS,
        "best_encoder_by_validation_dice": best_encoder,
        "best_threshold": best_threshold,
        "training_device_required": "cuda",
    }
    (RESULTS_ROOT / "execution_manifest.json").write_text(json.dumps(execution_manifest, indent=2), encoding="utf-8")
    shutil.copy2(config_path, RESULTS_ROOT / "kaggle_research_v1_2_runtime.yaml")
    run(
        [
            sys.executable,
            "scripts/package_release_artifacts.py",
            "--source-root",
            WORKING_ROOT,
            "--output-dir",
            WORKING_ROOT / "release_artifacts",
            "--package-name",
            "medical-segmentation-research-artifacts-v1.2",
        ],
        cwd=REPOSITORY_ROOT,
    )
    shutil.rmtree(PREPARED_ROOT)
    print("Research v1.2 workflow completed:", RESULTS_ROOT, flush=True)


if __name__ == "__main__":
    main()
