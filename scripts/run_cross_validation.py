import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cross_validation import (  # noqa: E402
    create_kfold_splits,
    materialize_fold_directories,
    paired_stems,
    write_folds,
)
from src.experiment_suite import summarize_seed_metrics, write_csv  # noqa: E402
from src.utils import load_config  # noqa: E402


def run(command, cwd):
    print(">>>", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def read_single_row(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_cross_validation_report(path, rows, summary, best_fold):
    lines = [
        "# Cross-Validation Summary",
        "",
        "This report summarizes validation metrics across materialized folds. Standard deviation uses `ddof=1` when at least two folds are available.",
        "",
        f"Best fold by validation Dice: `{best_fold}`",
        "",
        "| Split | Metric | Folds | Mean | Std | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append(
            "| {split} | {metric} | {runs} | {mean:.6f} | {std:.6f} | {min:.6f} | {max:.6f} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Per-Fold Validation Results",
            "",
            "| Fold | Dice | IoU | Precision | Recall | Loss | Checkpoint |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: int(item["fold"])):
        lines.append(
            f"| {row['fold']} | {float(row['dice']):.6f} | {float(row['iou']):.6f} | "
            f"{float(row['precision']):.6f} | {float(row['recall']):.6f} | "
            f"{float(row['loss']):.6f} | `{row['checkpoint_path']}` |"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation training on Kaggle/local paths.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-root", default="outputs/cross_validation")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stems = paired_stems(args.images_dir, args.masks_dir)
    folds = create_kfold_splits(stems, k=args.k, seed=args.seed)
    write_folds(output_root / "folds.json", folds, metadata={"k": args.k, "seed": args.seed, "samples": len(stems)})
    fold_data_root = output_root / "fold_data"
    materialize_fold_directories(args.images_dir, args.masks_dir, folds, fold_data_root)

    runtime_config_dir = output_root / "runtime_configs"
    runtime_config_dir.mkdir(exist_ok=True)
    rows = []
    for fold in folds:
        fold_index = int(fold["fold"])
        fold_root = output_root / f"fold_{fold_index}"
        fold_config = json.loads(json.dumps(config))
        fold_config["seed"] = int(args.seed) + fold_index
        fold_config["experiment_name"] = f"{config.get('experiment_name', 'experiment')}_fold_{fold_index}"
        fold_config.setdefault("data", {})
        fold_config["data"]["train_images_dir"] = str(fold_data_root / f"fold_{fold_index}/train/images")
        fold_config["data"]["train_masks_dir"] = str(fold_data_root / f"fold_{fold_index}/train/masks")
        fold_config["data"]["val_images_dir"] = str(fold_data_root / f"fold_{fold_index}/val/images")
        fold_config["data"]["val_masks_dir"] = str(fold_data_root / f"fold_{fold_index}/val/masks")
        fold_config.setdefault("paths", {})
        fold_config["paths"]["output_dir"] = str(fold_root / "outputs")
        fold_config["paths"]["checkpoint_dir"] = str(fold_root / "checkpoints")
        if args.epochs is not None:
            fold_config.setdefault("training", {})["epochs"] = int(args.epochs)
        if args.patience is not None:
            fold_config.setdefault("training", {}).setdefault("early_stopping", {})["patience"] = int(args.patience)
        config_path = runtime_config_dir / f"fold_{fold_index}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")

        run([sys.executable, "train.py", "--config", config_path], cwd=ROOT)
        history_path = fold_root / "outputs/metrics.csv"
        if history_path.exists():
            history_path.replace(fold_root / "outputs/training_history.csv")
        checkpoint_path = fold_root / "checkpoints/best_model.pth"
        run(
            [
                sys.executable,
                "evaluate.py",
                "--config",
                config_path,
                "--checkpoint",
                checkpoint_path,
                "--split",
                "val",
            ],
            cwd=ROOT,
        )
        metrics_path = fold_root / "outputs/metrics.csv"
        row = read_single_row(metrics_path)
        row = {
            "seed": fold_config["seed"],
            "fold": fold_index,
            "split": "val",
            "checkpoint_path": str(checkpoint_path),
            **row,
        }
        rows.append(row)
        write_csv(fold_root / "outputs/fold_val_metrics.csv", [row])

    summary = summarize_seed_metrics(rows)
    write_csv(output_root / "cross_validation_metrics.csv", rows)
    write_csv(output_root / "cross_validation_summary.csv", summary)
    best_fold = max(rows, key=lambda row: float(row["dice"]))["fold"]
    write_cross_validation_report(output_root / "cross_validation_summary.md", rows, summary, best_fold)
    print(f"Cross-validation complete. Best fold by validation Dice: {best_fold}")


if __name__ == "__main__":
    main()
