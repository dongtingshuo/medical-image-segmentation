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


def write_encoder_report(path, rows, summary, best_encoder):
    lines = [
        "# Encoder Comparison Summary",
        "",
        "This report compares high-capacity segmentation backbones under the same dataset split and training recipe.",
        "",
        f"Best encoder by validation Dice: `{best_encoder}`",
        "",
        "| Encoder | Dice | IoU | Precision | Recall | Loss | Checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda item: item["encoder"]):
        lines.append(
            f"| {row['encoder']} | {float(row['dice']):.6f} | {float(row['iou']):.6f} | "
            f"{float(row['precision']):.6f} | {float(row['recall']):.6f} | "
            f"{float(row['loss']):.6f} | `{row['checkpoint_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Metric Summary",
            "",
            "| Split | Metric | Runs | Mean | Std | Min | Max |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary:
        lines.append(
            "| {split} | {metric} | {runs} | {mean:.6f} | {std:.6f} | {min:.6f} | {max:.6f} |".format(
                **item
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Compare encoder backbones with a shared training recipe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs/encoder_comparison")
    parser.add_argument("--model-name", default="unet_plus_plus")
    parser.add_argument("--encoders", nargs="+", default=["efficientnet-b3", "resnet34"])
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_config_dir = output_root / "runtime_configs"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for encoder in args.encoders:
        encoder_key = str(encoder).replace("/", "_")
        encoder_root = output_root / encoder_key
        run_config = json.loads(json.dumps(config))
        run_config["experiment_name"] = f"{config.get('experiment_name', 'experiment')}_{encoder_key}"
        run_config.setdefault("model", {})
        run_config["model"]["model_name"] = args.model_name
        run_config["model"]["encoder_name"] = encoder
        run_config["model"]["encoder_weights"] = args.encoder_weights
        run_config.setdefault("paths", {})
        run_config["paths"]["output_dir"] = str(encoder_root / "outputs")
        run_config["paths"]["checkpoint_dir"] = str(encoder_root / "checkpoints")
        if args.epochs is not None:
            run_config.setdefault("training", {})["epochs"] = int(args.epochs)
        if args.patience is not None:
            run_config.setdefault("training", {}).setdefault("early_stopping", {})["patience"] = int(args.patience)
        if args.batch_size is not None:
            run_config.setdefault("training", {})["batch_size"] = int(args.batch_size)
        if args.image_size is not None:
            run_config.setdefault("data", {})["image_size"] = int(args.image_size)

        config_path = runtime_config_dir / f"{encoder_key}.yaml"
        config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        run([sys.executable, "train.py", "--config", config_path], cwd=ROOT)
        history_path = encoder_root / "outputs/metrics.csv"
        if history_path.exists():
            history_path.replace(encoder_root / "outputs/training_history.csv")
        checkpoint_path = encoder_root / "checkpoints/best_model.pth"
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
        row = read_single_row(encoder_root / "outputs/metrics.csv")
        row = {"encoder": encoder, "split": "val", "checkpoint_path": str(checkpoint_path), **row}
        rows.append(row)
        write_csv(encoder_root / "outputs/encoder_val_metrics.csv", [row])

    summary = summarize_seed_metrics(rows)
    best_encoder = max(rows, key=lambda row: float(row["dice"]))["encoder"]
    write_csv(output_root / "encoder_comparison_metrics.csv", rows)
    write_csv(output_root / "encoder_comparison_summary.csv", summary)
    write_encoder_report(output_root / "encoder_comparison_summary.md", rows, summary, best_encoder)
    manifest = {
        "best_encoder_by_validation_dice": best_encoder,
        "encoders": args.encoders,
        "model_name": args.model_name,
    }
    (output_root / "encoder_comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Encoder comparison complete. Best encoder by validation Dice: {best_encoder}")


if __name__ == "__main__":
    main()
