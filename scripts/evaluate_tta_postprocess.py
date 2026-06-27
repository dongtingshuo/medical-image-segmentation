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
from src.metrics import (  # noqa: E402
    boundary_f1_score,
    dice_score,
    iou_score,
    precision_score,
    recall_score,
    specificity_score,
)
from src.postprocessing import postprocess_binary_masks, predict_probabilities_tta  # noqa: E402
from src.threshold_search import compute_threshold_metrics_from_counts  # noqa: E402
from src.utils import (  # noqa: E402
    checkpoint_model_config,
    create_dirs,
    get_device,
    load_checkpoint,
    load_checkpoint_payload,
    load_config,
)


def parse_scales(value):
    scales = [float(item) for item in str(value).replace(" ", "").split(",") if item]
    if not scales:
        raise ValueError("At least one TTA scale is required.")
    invalid = [scale for scale in scales if scale <= 0]
    if invalid:
        raise ValueError(f"TTA scales must be positive, got {invalid}")
    return scales


def _logits_from_probabilities(probabilities):
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probabilities)


def _logits_from_binary_masks(masks):
    return masks.float() * 20.0 - 10.0


def _update_counts(preds, masks, counts):
    true = masks > 0.5
    pred = preds > 0.5
    counts["tp"] += float((pred & true).sum().item())
    counts["fp"] += float((pred & ~true).sum().item())
    counts["fn"] += float((~pred & true).sum().item())
    counts["tn"] += float((~pred & ~true).sum().item())
    counts["pixels"] += int(true.numel())
    counts["samples"] += int(true.shape[0]) if true.ndim >= 3 else 1


@torch.no_grad()
def evaluate_tta_postprocess(
    model,
    loader,
    device,
    threshold=0.5,
    scales=(1.0,),
    horizontal_flip=False,
    vertical_flip=False,
    min_component_area=0,
    fill_holes=False,
):
    model.eval()
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "pixels": 0, "samples": 0}
    metric_totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "boundary_f1": 0.0,
    }
    sample_total = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        probabilities = predict_probabilities_tta(
            model,
            images,
            scales=scales,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        )
        preds = (probabilities >= threshold).float()
        if min_component_area or fill_holes:
            preds = postprocess_binary_masks(preds, min_component_area=min_component_area, fill_holes=fill_holes)
            boundary_logits = _logits_from_binary_masks(preds)
        else:
            boundary_logits = _logits_from_probabilities(probabilities)
        _update_counts(preds.detach().cpu(), masks.detach().cpu(), counts)
        batch_size = int(images.shape[0])
        batch_metrics = {
            "dice": dice_score(boundary_logits, masks, threshold=threshold).item(),
            "iou": iou_score(boundary_logits, masks, threshold=threshold).item(),
            "precision": precision_score(boundary_logits, masks, threshold=threshold).item(),
            "recall": recall_score(boundary_logits, masks, threshold=threshold).item(),
            "specificity": specificity_score(boundary_logits, masks, threshold=threshold).item(),
            "boundary_f1": boundary_f1_score(boundary_logits, masks, threshold=threshold).item(),
        }
        for key, value in batch_metrics.items():
            metric_totals[key] += float(value) * batch_size
        sample_total += batch_size

    metrics = {key: value / max(sample_total, 1) for key, value in metric_totals.items()}
    micro_metrics = compute_threshold_metrics_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
    metrics.update({f"micro_{key}": value for key, value in micro_metrics.items()})
    metrics.update(
        {
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
    csv_path = output_dir / "tta_postprocess_metrics.csv"
    json_path = output_dir / "tta_postprocess_metrics.json"
    md_path = output_dir / "tta_postprocess_metrics.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    lines = [
        "# TTA and Post-Processing Evaluation",
        "",
        f"- Split: `{settings['split']}`",
        f"- Threshold: `{settings['threshold']:.3f}`",
        f"- Scales: `{settings['scales']}`",
        f"- Horizontal flip: `{settings['horizontal_flip']}`",
        f"- Vertical flip: `{settings['vertical_flip']}`",
        f"- Min component area: `{settings['min_component_area']}`",
        f"- Fill holes: `{settings['fill_holes']}`",
        f"- Batch size: `{settings['batch_size']}`",
        "- Primary aggregation: `macro_per_image`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in ["dice", "iou", "precision", "recall", "specificity", "boundary_f1"]:
        lines.append(f"| {key} | {metrics[key]:.6f} |")
    for key in ["micro_dice", "micro_iou", "micro_precision", "micro_recall", "micro_specificity"]:
        lines.append(f"| {key} | {metrics[key]:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation TTA and mask post-processing.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test", "external"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--vertical-flip", action="store_true")
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--output-dir", default="outputs/tta_postprocess")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {args.threshold}")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    scales = parse_scales(args.scales)
    config = load_config(args.config)
    images_path, masks_path = resolve_split_paths(config, args.split)
    device = get_device(args.device or config.get("device", "auto"))
    dataset = SkinLesionDataset(images_path, masks_path, transform=get_val_transforms(config))
    batch_size = args.batch_size or int(config.get("training", {}).get("batch_size", 8))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    checkpoint = load_checkpoint_payload(args.checkpoint, device=device)
    model = build_model_from_config(config, checkpoint=checkpoint).to(device)
    expected_model_config = checkpoint_model_config(checkpoint) or config.get("model", {})
    load_checkpoint(args.checkpoint, model, device, expected_model_config=expected_model_config, checkpoint=checkpoint)
    metrics = evaluate_tta_postprocess(
        model,
        loader,
        device,
        threshold=args.threshold,
        scales=scales,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        min_component_area=args.min_component_area,
        fill_holes=args.fill_holes,
    )
    settings = {
        "split": args.split,
        "threshold": args.threshold,
        "scales": ",".join(f"{scale:g}" for scale in scales),
        "horizontal_flip": bool(args.horizontal_flip),
        "vertical_flip": bool(args.vertical_flip),
        "min_component_area": int(args.min_component_area),
        "fill_holes": bool(args.fill_holes),
        "batch_size": int(batch_size),
        "aggregation": "macro_per_image",
    }
    outputs = write_outputs(metrics, args.output_dir, settings)
    print(f"Saved TTA/post-processing metrics to {outputs['csv']}")
    for key in ["dice", "iou", "precision", "recall", "specificity", "boundary_f1"]:
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
