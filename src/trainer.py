import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.experiment import append_experiment_result
from src.metrics import dice_score, iou_score, precision_score, recall_score
from src.utils import format_time, save_checkpoint
from src.visualization import plot_training_curves, save_sample_predictions


def _metric_dict(logits, masks):
    return {
        "dice": dice_score(logits, masks).item(),
        "iou": iou_score(logits, masks).item(),
        "precision": precision_score(logits, masks).item(),
        "recall": recall_score(logits, masks).item(),
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False, max_batches=None):
    model.train()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0
    progress = tqdm(dataloader, desc="train", leave=False)
    for batch_idx, (images, masks) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        metrics = _metric_dict(logits.detach(), masks)
        totals["loss"] += loss.item()
        for key, value in metrics.items():
            totals[key] += value
        count += 1
        progress.set_postfix(loss=totals["loss"] / count, dice=totals["dice"] / count)
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, max_batches=None):
    model.eval()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0
    progress = tqdm(dataloader, desc="val", leave=False)
    for batch_idx, (images, masks) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        metrics = _metric_dict(logits, masks)
        totals["loss"] += loss.item()
        for key, value in metrics.items():
            totals[key] += value
        count += 1
        progress.set_postfix(loss=totals["loss"] / count, dice=totals["dice"] / count)
    return {key: value / max(count, 1) for key, value in totals.items()}


def _is_improved(value, best, mode):
    return value > best if mode == "max" else value < best


def _save_metrics_csv(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"
    fields = list(history.keys())
    rows = zip(*[history[field] for field in fields])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    config,
):
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    output_dir = Path(paths_cfg.get("output_dir", "outputs"))
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    curves_dir = output_dir / "curves"
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(training_cfg.get("epochs", 1))
    use_amp = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 10))
    monitor = early_cfg.get("monitor", "val_dice")
    mode = early_cfg.get("mode", "max")
    best_score = -float("inf") if mode == "max" else float("inf")
    bad_epochs = 0
    best_metrics = {}
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"
    start = time.time()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
    }

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=training_cfg.get("max_train_batches"),
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=training_cfg.get("max_val_batches"),
        )

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        print(
            "train_loss={:.4f} val_loss={:.4f} dice={:.4f} iou={:.4f} precision={:.4f} recall={:.4f}".format(
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"],
            )
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        }
        save_checkpoint(state, last_path)

        monitored_value = val_metrics[monitor.replace("val_", "")]
        if _is_improved(monitored_value, best_score, mode):
            best_score = monitored_value
            bad_epochs = 0
            best_metrics = val_metrics
            save_checkpoint(state, best_path)
            print(f"Saved best checkpoint to {best_path}")
        else:
            bad_epochs += 1
            if early_enabled and bad_epochs >= patience:
                print(f"Early stopping triggered after {bad_epochs} non-improving epochs.")
                break

    training_time = format_time(time.time() - start)
    _save_metrics_csv(history, output_dir)
    plot_training_curves(history, curves_dir / "training_curves.png")
    save_sample_predictions(model, val_loader, device, samples_dir, max_samples=4)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    result = {
        "experiment_name": config.get("experiment_name", ""),
        "model_name": model_cfg.get("model_name", ""),
        "device": str(device),
        "image_size": data_cfg.get("image_size", ""),
        "batch_size": training_cfg.get("batch_size", ""),
        "epochs": training_cfg.get("epochs", ""),
        "lr": training_cfg.get("lr", ""),
        "loss_name": training_cfg.get("loss_name", ""),
        "augmentation_enabled": config.get("augmentation", {}).get("enabled", ""),
        "best_val_loss": best_metrics.get("loss", ""),
        "best_dice": best_metrics.get("dice", ""),
        "best_iou": best_metrics.get("iou", ""),
        "precision": best_metrics.get("precision", ""),
        "recall": best_metrics.get("recall", ""),
        "checkpoint_path": str(best_path),
        "training_time": training_time,
    }
    append_experiment_result(output_dir, result)
    return {"history": history, "best_checkpoint": best_path, "last_checkpoint": last_path, "best_metrics": best_metrics}


fit = train_model

