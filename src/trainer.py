import csv
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from src.experiment import append_experiment_result
from src.metrics import boundary_f1_score, dice_score, iou_score, precision_score, recall_score, specificity_score
from src.utils import CHECKPOINT_FORMAT_VERSION, format_time, load_checkpoint, save_checkpoint
from src.visualization import plot_training_curves, save_sample_predictions


def _metric_dict(logits, masks):
    metrics = {
        "dice": dice_score(logits, masks).item(),
        "iou": iou_score(logits, masks).item(),
        "precision": precision_score(logits, masks).item(),
        "recall": recall_score(logits, masks).item(),
        "specificity": specificity_score(logits, masks).item(),
        "boundary_f1": boundary_f1_score(logits, masks).item(),
    }
    for name, value in metrics.items():
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError(f"Non-finite metric detected: {name}={value}")
    return metrics


def _ensure_finite_tensor(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {label} detected. Stop training and inspect data, loss, lr, and masks.")


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False, max_batches=None):
    model.train()
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
    progress = tqdm(dataloader, desc="train", leave=False)
    for batch_idx, (images, masks) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        _ensure_finite_tensor(loss, "training loss")
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        metrics = _metric_dict(logits.detach(), masks)
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        for key, value in metrics.items():
            totals[key] += value * batch_size
        count += batch_size
        progress.set_postfix(loss=totals["loss"] / count, dice=totals["dice"] / count)
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, max_batches=None):
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
    progress = tqdm(dataloader, desc="val", leave=False)
    for batch_idx, (images, masks) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        _ensure_finite_tensor(loss, "validation loss")
        metrics = _metric_dict(logits, masks)
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        for key, value in metrics.items():
            totals[key] += value * batch_size
        count += batch_size
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 10))
    monitor = early_cfg.get("monitor", "val_dice")
    mode = early_cfg.get("mode", "max")
    best_score = -float("inf") if mode == "max" else float("inf")
    bad_epochs = 0
    best_metrics = {}
    best_epoch = None
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
        "val_specificity": [],
        "val_boundary_f1": [],
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
        history["val_specificity"].append(val_metrics["specificity"])
        history["val_boundary_f1"].append(val_metrics["boundary_f1"])

        print(
            "train_loss={:.4f} val_loss={:.4f} dice={:.4f} iou={:.4f} precision={:.4f} "
            "recall={:.4f} specificity={:.4f} boundary_f1={:.4f}".format(
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["specificity"],
                val_metrics["boundary_f1"],
            )
        )

        state = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
            "metadata": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "monitor": monitor,
                "monitor_mode": mode,
            },
        }
        save_checkpoint(state, last_path)

        monitor_key = monitor.replace("val_", "")
        if monitor_key not in val_metrics:
            raise KeyError(f"Unsupported early stopping monitor `{monitor}`. Available: {sorted(val_metrics)}")
        monitored_value = val_metrics[monitor_key]
        if not torch.isfinite(torch.tensor(monitored_value)):
            raise FloatingPointError(f"Non-finite monitored metric detected: {monitor}={monitored_value}")
        if _is_improved(monitored_value, best_score, mode):
            best_score = monitored_value
            bad_epochs = 0
            best_metrics = val_metrics
            best_epoch = epoch
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
    load_checkpoint(
        best_path,
        model,
        device,
        expected_model_config=config.get("model", {}),
    )
    save_sample_predictions(model, val_loader, device, samples_dir, max_samples=4)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    result = {
        "experiment_name": config.get("experiment_name", ""),
        "model_name": model_cfg.get("model_name", ""),
        "device": str(device),
        "image_size": data_cfg.get("image_size", ""),
        "batch_size": training_cfg.get("batch_size", ""),
        "epochs": len(history["epoch"]),
        "requested_epochs": training_cfg.get("epochs", ""),
        "best_epoch": best_epoch,
        "lr": training_cfg.get("lr", ""),
        "loss_name": training_cfg.get("loss_name", ""),
        "augmentation_enabled": config.get("augmentation", {}).get("enabled", ""),
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else "",
        "val_loss_at_best_epoch": best_metrics.get("loss", ""),
        "best_dice": best_metrics.get("dice", ""),
        "best_iou": best_metrics.get("iou", ""),
        "precision": best_metrics.get("precision", ""),
        "recall": best_metrics.get("recall", ""),
        "specificity": best_metrics.get("specificity", ""),
        "boundary_f1": best_metrics.get("boundary_f1", ""),
        "checkpoint_path": str(best_path),
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "training_time": training_time,
    }
    append_experiment_result(output_dir, result)
    return {"history": history, "best_checkpoint": best_path, "last_checkpoint": last_path, "best_metrics": best_metrics}


fit = train_model
