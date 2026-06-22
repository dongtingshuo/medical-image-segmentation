import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import SkinLesionDataset, get_val_transforms
from src.inference import build_model_from_config
from src.losses import build_loss
from src.metrics import boundary_f1_score, dice_score, iou_score, precision_score, recall_score, specificity_score
from src.utils import (
    checkpoint_model_config,
    create_dirs,
    data_path,
    get_device,
    load_checkpoint,
    load_checkpoint_payload,
    load_config,
)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "boundary_f1": 0.0,
    }
    count = 0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite evaluation loss detected. Check checkpoint, data, and masks.")
        batch_metrics = {
            "loss": loss.item(),
            "dice": dice_score(logits, masks, threshold=threshold).item(),
            "iou": iou_score(logits, masks, threshold=threshold).item(),
            "precision": precision_score(logits, masks, threshold=threshold).item(),
            "recall": recall_score(logits, masks, threshold=threshold).item(),
            "specificity": specificity_score(logits, masks, threshold=threshold).item(),
            "boundary_f1": boundary_f1_score(logits, masks, threshold=threshold).item(),
        }
        batch_size = images.size(0)
        for key, value in batch_metrics.items():
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError(f"Non-finite evaluation metric detected: {key}={value}")
            totals[key] += value * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {args.threshold}")

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    images_value = data_path(config, f"{args.split}_images_dir")
    masks_value = data_path(config, f"{args.split}_masks_dir")
    if not images_value or not masks_value:
        raise ValueError(
            f"Config does not define {args.split}_images_dir and {args.split}_masks_dir. "
            f"Add them under the `data` section before evaluating the {args.split} split."
        )
    images_path = Path(images_value)
    masks_path = Path(masks_value)
    if not images_path.exists() or not masks_path.exists():
        raise FileNotFoundError(f"{args.split} paths do not exist: images={images_path}, masks={masks_path}")

    device = get_device(config.get("device", "auto"))
    dataset = SkinLesionDataset(images_path, masks_path, transform=get_val_transforms(config))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    checkpoint_payload = load_checkpoint_payload(checkpoint, device=device)
    model = build_model_from_config(config, checkpoint=checkpoint_payload).to(device)
    expected_model_config = checkpoint_model_config(checkpoint_payload) or config.get("model", {})
    load_checkpoint(
        checkpoint,
        model,
        device,
        expected_model_config=expected_model_config,
        checkpoint=checkpoint_payload,
    )
    criterion = build_loss(config)
    metrics = evaluate(model, loader, criterion, device, threshold=args.threshold)
    metrics = {"split": args.split, "samples": len(dataset), "threshold": args.threshold, **metrics}

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    create_dirs(output_dir)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("Evaluation results")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
