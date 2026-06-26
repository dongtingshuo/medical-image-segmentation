import csv
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from src.experiment import append_experiment_result
from src.metrics import boundary_f1_score, dice_score, iou_score, precision_score, recall_score, specificity_score
from src.utils import CHECKPOINT_FORMAT_VERSION, format_time, load_checkpoint, load_checkpoint_payload, save_checkpoint
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


def _empty_history():
    return {
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


def _history_from_checkpoint(checkpoint):
    history = _empty_history()
    saved_history = checkpoint.get("history", {})
    if isinstance(saved_history, dict):
        for key in history:
            value = saved_history.get(key, [])
            history[key] = list(value) if isinstance(value, (list, tuple)) else []
    return history


def _score_from_metrics(metrics, monitor):
    monitor_key = monitor.replace("val_", "")
    if not isinstance(metrics, dict) or monitor_key not in metrics:
        return None
    return float(metrics[monitor_key])


def _load_resume_state(
    resume_path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    expected_model_config,
    monitor,
    default_best_score,
):
    checkpoint = load_checkpoint_payload(resume_path, device=device)
    load_checkpoint(
        resume_path,
        model,
        device,
        optimizer=optimizer,
        expected_model_config=expected_model_config,
        checkpoint=checkpoint,
    )
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"Scheduler state was not restored: {exc}", RuntimeWarning, stacklevel=2)
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"AMP scaler state was not restored: {exc}", RuntimeWarning, stacklevel=2)

    best_metrics = checkpoint.get("best_metrics", {})
    best_score = checkpoint.get("best_score")
    if best_score is None:
        best_score = _score_from_metrics(best_metrics, monitor)
    if best_score is None:
        best_score = _score_from_metrics(checkpoint.get("val_metrics", {}), monitor)
    return {
        "start_epoch": int(checkpoint.get("epoch", 0)) + 1,
        "history": _history_from_checkpoint(checkpoint),
        "best_score": default_best_score if best_score is None else float(best_score),
        "bad_epochs": int(checkpoint.get("bad_epochs", 0)),
        "best_metrics": best_metrics if isinstance(best_metrics, dict) else {},
        "best_epoch": checkpoint.get("best_epoch"),
        "checkpoint": checkpoint,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    config,
    resume_path=None,
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

    history = _empty_history()
    start_epoch = 1
    if resume_path is not None:
        resume_state = _load_resume_state(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_model_config=config.get("model", {}),
            monitor=monitor,
            default_best_score=best_score,
        )
        start_epoch = resume_state["start_epoch"]
        history = resume_state["history"]
        best_score = resume_state["best_score"]
        bad_epochs = resume_state["bad_epochs"]
        best_metrics = resume_state["best_metrics"]
        best_epoch = resume_state["best_epoch"]
        print(f"Resuming training from {resume_path} at epoch {start_epoch}/{epochs}.")

        if not best_metrics and best_path.exists():
            try:
                best_checkpoint = load_checkpoint_payload(best_path, device=device)
                best_metrics = best_checkpoint.get("best_metrics", best_checkpoint.get("val_metrics", {}))
                best_epoch = best_checkpoint.get("best_epoch", best_checkpoint.get("epoch", best_epoch))
                restored_score = _score_from_metrics(best_metrics, monitor)
                if restored_score is not None:
                    best_score = restored_score
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Existing best checkpoint was not inspected during resume: {exc}", RuntimeWarning, stacklevel=2)

    for epoch in range(start_epoch, epochs + 1):
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
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "val_metrics": val_metrics,
            "history": history,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "metadata": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "monitor": monitor,
                "monitor_mode": mode,
            },
        }

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
            state["best_score"] = best_score
            state["best_metrics"] = best_metrics
            state["best_epoch"] = best_epoch
            state["bad_epochs"] = bad_epochs
            save_checkpoint(state, best_path)
            print(f"Saved best checkpoint to {best_path}")
        else:
            bad_epochs += 1
            state["bad_epochs"] = bad_epochs
            if early_enabled and bad_epochs >= patience:
                print(f"Early stopping triggered after {bad_epochs} non-improving epochs.")
                save_checkpoint(state, last_path)
                break
        save_checkpoint(state, last_path)

    training_time = format_time(time.time() - start)
    _save_metrics_csv(history, output_dir)
    plot_training_curves(history, curves_dir / "training_curves.png")
    checkpoint_for_samples = best_path if best_path.exists() else resume_path
    if checkpoint_for_samples is None:
        raise FileNotFoundError("No best checkpoint is available for sample prediction.")
    load_checkpoint(
        checkpoint_for_samples,
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
