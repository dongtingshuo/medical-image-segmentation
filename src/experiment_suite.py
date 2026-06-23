import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

EVALUATION_METRICS = (
    "loss",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "boundary_f1",
)


def read_single_row_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected result file does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def summarize_seed_metrics(rows, metrics=EVALUATION_METRICS):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["split"], []).append(row)

    summary = []
    for split, split_rows in sorted(grouped.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in split_rows if row.get(metric) not in {None, ""}]
            if not values:
                continue
            summary.append(
                {
                    "split": split,
                    "metric": metric,
                    "runs": len(values),
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )
    return summary


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer CSV columns for empty rows: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    return path


def write_summary_markdown(path, rows, summary, best_seed):
    path = Path(path)
    splits = sorted({row["split"] for row in rows})
    lines = [
        "# Repeated Experiment Summary",
        "",
        "Three-run variability uses the sample standard deviation (`ddof=1`).",
        "",
        f"Best seed by validation Dice: `{best_seed}`",
        "",
        "| Split | Metric | Runs | Mean | Std | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append(
            "| {split} | {metric} | {runs} | {mean:.6f} | {std:.6f} | {min:.6f} | {max:.6f} |".format(
                **item
            )
        )
    lines.extend(["", "Evaluated splits: " + ", ".join(f"`{split}`" for split in splits), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_command(command, cwd):
    print(">>>", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def _evaluate_split(repo_root, config_path, checkpoint_path, split, output_path):
    run_command(
        [
            sys.executable,
            "evaluate.py",
            "--config",
            config_path,
            "--checkpoint",
            checkpoint_path,
            "--split",
            split,
        ],
        cwd=repo_root,
    )
    generated = Path(yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["paths"]["output_dir"]) / "metrics.csv"
    row = read_single_row_csv(generated)
    write_csv(output_path, [row])
    return row


def run_repeated_experiments(
    config,
    seeds,
    repo_root,
    output_root,
    test_images_dir=None,
    test_masks_dir=None,
    external_images_dir=None,
    external_masks_dir=None,
):
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) < 3:
        raise ValueError("Repeated experiments require at least three distinct random seeds.")
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_config_dir = output_root / "runtime_configs"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    if bool(test_images_dir) != bool(test_masks_dir):
        raise ValueError("Both test_images_dir and test_masks_dir must be provided together.")
    if bool(external_images_dir) != bool(external_masks_dir):
        raise ValueError("Both external_images_dir and external_masks_dir must be provided together.")

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        output_dir = seed_root / "outputs"
        checkpoint_dir = seed_root / "checkpoints"
        seed_config = json.loads(json.dumps(config))
        seed_config["seed"] = seed
        seed_config["experiment_name"] = f"{config.get('experiment_name', 'experiment')}_seed_{seed}"
        seed_config.setdefault("paths", {})["output_dir"] = str(output_dir)
        seed_config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
        seed_config.setdefault("data", {})
        if test_images_dir:
            seed_config["data"]["test_images_dir"] = str(Path(test_images_dir).resolve())
            seed_config["data"]["test_masks_dir"] = str(Path(test_masks_dir).resolve())

        config_path = runtime_config_dir / f"seed_{seed}.yaml"
        config_path.write_text(yaml.safe_dump(seed_config, sort_keys=False), encoding="utf-8")
        run_command([sys.executable, "train.py", "--config", config_path], cwd=repo_root)

        history_path = output_dir / "metrics.csv"
        history_path.replace(output_dir / "training_history.csv")
        checkpoint_path = checkpoint_dir / "best_model.pth"

        for split in ("val", "test"):
            if split == "test" and not seed_config["data"].get("test_images_dir"):
                continue
            row = _evaluate_split(
                repo_root,
                config_path,
                checkpoint_path,
                split,
                output_dir / f"{split}_metrics.csv",
            )
            all_rows.append({"seed": seed, "split": split, **row})

        if external_images_dir:
            external_config = json.loads(json.dumps(seed_config))
            external_config["data"]["test_images_dir"] = str(Path(external_images_dir).resolve())
            external_config["data"]["test_masks_dir"] = str(Path(external_masks_dir).resolve())
            external_config_path = runtime_config_dir / f"seed_{seed}_external.yaml"
            external_config_path.write_text(yaml.safe_dump(external_config, sort_keys=False), encoding="utf-8")
            row = _evaluate_split(
                repo_root,
                external_config_path,
                checkpoint_path,
                "test",
                output_dir / "external_metrics.csv",
            )
            row["split"] = "external"
            write_csv(output_dir / "external_metrics.csv", [row])
            all_rows.append({"seed": seed, **row})

    summary = summarize_seed_metrics(all_rows)
    validation_rows = [row for row in all_rows if row["split"] == "val"]
    if not validation_rows:
        raise RuntimeError("No validation metrics were produced.")
    best_seed = max(validation_rows, key=lambda row: float(row["dice"]))["seed"]
    write_csv(output_root / "all_seed_metrics.csv", all_rows)
    write_csv(output_root / "summary.csv", summary)
    write_summary_markdown(output_root / "summary.md", all_rows, summary, best_seed)
    manifest = {
        "seeds": seeds,
        "best_seed_by_validation_dice": int(best_seed),
        "test_split_evaluated": bool(test_images_dir or config.get("data", {}).get("test_images_dir")),
        "external_split_evaluated": bool(external_images_dir),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"rows": all_rows, "summary": summary, "best_seed": int(best_seed)}
