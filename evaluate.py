import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import SkinLesionDataset, get_val_transforms
from src.losses import get_loss
from src.metrics import dice_score, iou_score, precision_score, recall_score
from src.inference import build_model_from_config
from src.utils import create_dirs, data_path, get_device, load_checkpoint, load_config


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
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
            "dice": dice_score(logits, masks).item(),
            "iou": iou_score(logits, masks).item(),
            "precision": precision_score(logits, masks).item(),
            "recall": recall_score(logits, masks).item(),
        }
        for key, value in batch_metrics.items():
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError(f"Non-finite evaluation metric detected: {key}={value}")
            totals[key] += value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    val_images = Path(data_path(config, "val_images_dir"))
    val_masks = Path(data_path(config, "val_masks_dir"))
    if not val_images.exists() or not val_masks.exists():
        raise FileNotFoundError(f"Validation paths do not exist: images={val_images}, masks={val_masks}")

    device = get_device(config.get("device", "auto"))
    dataset = SkinLesionDataset(val_images, val_masks, transform=get_val_transforms(config))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    model = build_model_from_config(config).to(device)
    load_checkpoint(checkpoint, model, device)
    criterion = get_loss(config.get("training", {}).get("loss_name", "bce_dice"))
    metrics = evaluate(model, loader, criterion, device)

    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs"))
    create_dirs(output_dir)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("Evaluation results")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
