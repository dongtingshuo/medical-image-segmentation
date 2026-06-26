import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import resolve_split_paths  # noqa: E402
from src.dataset import SkinLesionDataset, get_val_transforms  # noqa: E402
from src.inference import build_model_from_config  # noqa: E402
from src.metrics import boundary_f1_score  # noqa: E402
from src.threshold_search import compute_threshold_metrics_from_counts  # noqa: E402
from src.utils import checkpoint_model_config, create_dirs, get_device, load_checkpoint, load_checkpoint_payload, load_config  # noqa: E402


def parse_member(value):
    if ":" not in value:
        raise ValueError("Ensemble members must use CONFIG:CHECKPOINT format.")
    config_path, checkpoint_path = value.split(":", maxsplit=1)
    return Path(config_path), Path(checkpoint_path)


def load_member(config_path, checkpoint_path, device):
    config = load_config(config_path)
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
    model = build_model_from_config(config, checkpoint=checkpoint).to(device)
    expected_model_config = checkpoint_model_config(checkpoint) or config.get("model", {})
    load_checkpoint(checkpoint_path, model, device, expected_model_config=expected_model_config, checkpoint=checkpoint)
    model.eval()
    return {"config": config, "model": model, "checkpoint": str(checkpoint_path)}


def _logits_from_probabilities(probabilities):
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probabilities)


@torch.no_grad()
def evaluate_ensemble(members, loader, device, threshold=0.5):
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "pixels": 0, "samples": 0}
    boundary_total = 0.0
    sample_total = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        probabilities = []
        for member in members:
            probabilities.append(torch.sigmoid(member["model"](images)))
        mean_probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
        preds = mean_probabilities >= threshold
        true = masks > 0.5
        counts["tp"] += float((preds & true).sum().item())
        counts["fp"] += float((preds & ~true).sum().item())
        counts["fn"] += float((~preds & true).sum().item())
        counts["tn"] += float((~preds & ~true).sum().item())
        counts["pixels"] += int(true.numel())
        counts["samples"] += int(true.shape[0]) if true.ndim >= 3 else 1
        batch_size = int(images.shape[0])
        boundary_total += float(
            boundary_f1_score(_logits_from_probabilities(mean_probabilities), masks, threshold=threshold).item()
        ) * batch_size
        sample_total += batch_size
    metrics = compute_threshold_metrics_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
    metrics.update(
        {
            "boundary_f1": boundary_total / max(sample_total, 1),
            "samples": counts["samples"],
            "pixels": counts["pixels"],
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "tn": int(counts["tn"]),
        }
    )
    return metrics


def write_outputs(metrics, output_dir, settings):
    output_dir = Path(output_dir)
    create_dirs(output_dir)
    row = {**settings, **metrics}
    csv_path = output_dir / "ensemble_metrics.csv"
    json_path = output_dir / "ensemble_metrics.json"
    md_path = output_dir / "ensemble_metrics.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    lines = [
        "# Ensemble Evaluation",
        "",
        f"- Split: `{settings['split']}`",
        f"- Threshold: `{settings['threshold']:.3f}`",
        f"- Members: `{settings['members']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in ["dice", "iou", "precision", "recall", "specificity", "boundary_f1"]:
        lines.append(f"| {key} | {metrics[key]:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def main():
    parser = argparse.ArgumentParser(description="Evaluate an average-probability checkpoint ensemble.")
    parser.add_argument("--member", action="append", required=True, help="CONFIG:CHECKPOINT. Repeat for each model.")
    parser.add_argument("--split", default="test", choices=["val", "test", "external"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/ensemble")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if len(args.member) < 2:
        raise ValueError("At least two ensemble members are required.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {args.threshold}")
    parsed_members = [parse_member(value) for value in args.member]
    base_config = load_config(parsed_members[0][0])
    images_path, masks_path = resolve_split_paths(base_config, args.split)
    device = get_device(args.device or base_config.get("device", "auto"))
    dataset = SkinLesionDataset(images_path, masks_path, transform=get_val_transforms(base_config))
    loader = DataLoader(
        dataset,
        batch_size=int(base_config.get("training", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(base_config.get("training", {}).get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    members = [load_member(config_path, checkpoint_path, device) for config_path, checkpoint_path in parsed_members]
    metrics = evaluate_ensemble(members, loader, device, threshold=args.threshold)
    settings = {
        "split": args.split,
        "threshold": args.threshold,
        "members": ";".join(member["checkpoint"] for member in members),
    }
    outputs = write_outputs(metrics, args.output_dir, settings)
    print(f"Saved ensemble metrics to {outputs['csv']}")
    for key in ["dice", "iou", "precision", "recall", "specificity", "boundary_f1"]:
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
