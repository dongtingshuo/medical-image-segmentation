import argparse
import copy
import math
from pathlib import Path

from torch.utils.data import DataLoader

from train import build_optimizer, build_scheduler
from src.dataset import SkinLesionDataset, get_train_transforms, get_val_transforms
from src.losses import get_loss
from src.model_factory import get_model
from src.trainer import train_model
from src.utils import create_dirs, data_path, get_device, load_config, set_seed
from src.visualization import save_sample_predictions


def write_report(path, config, device, result):
    history = result["history"]
    losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    values = losses + val_losses + history.get("val_dice", []) + history.get("val_iou", [])
    has_nan = any(math.isnan(float(v)) for v in values)
    loss_exploded = bool(losses and max(losses) > max(losses[0] * 10.0, 10.0))
    lines = [
        "# Quick Train Report",
        "",
        f"- Model: {config.get('model', {}).get('model_name', 'unet')}",
        f"- Device: {device}",
        f"- Epochs: {len(history.get('epoch', []))}",
        f"- Initial train loss: {losses[0] if losses else 'N/A'}",
        f"- Final train loss: {losses[-1] if losses else 'N/A'}",
        f"- Final val loss: {val_losses[-1] if val_losses else 'N/A'}",
        f"- Final Dice: {history.get('val_dice', ['N/A'])[-1]}",
        f"- Final IoU: {history.get('val_iou', ['N/A'])[-1]}",
        f"- NaN detected: {has_nan}",
        f"- Loss exploded: {loss_exploded}",
        f"- Best checkpoint: {result.get('best_checkpoint')}",
        "",
        "## Interpretation",
        "",
        "- Normal: training completes, metrics are finite, and loss does not explode.",
        "- Quick train is not expected to reach high accuracy.",
        "- If NaN appears or prediction samples are all black/all white, do not start formal long training.",
    ]
    Path(path).write_text("\n".join(map(str, lines)) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    config = copy.deepcopy(config)
    config.setdefault("training", {})
    config["training"]["epochs"] = max(1, min(args.epochs, 3))
    config["training"].setdefault("max_train_batches", 20)
    config["training"].setdefault("max_val_batches", 10)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))

    paths = config.get("paths", {})
    output_dir = Path(paths.get("output_dir", "outputs"))
    checkpoint_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    sanity_dir = output_dir / "sanity_check"
    create_dirs(output_dir, checkpoint_dir, sanity_dir)

    train_dataset = SkinLesionDataset(data_path(config, "train_images_dir"), data_path(config, "train_masks_dir"), get_train_transforms(config))
    val_dataset = SkinLesionDataset(data_path(config, "val_images_dir"), data_path(config, "val_masks_dir"), get_val_transforms(config))
    training = config.get("training", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model_cfg = dict(config.get("model", {}))
    model_name = model_cfg.pop("model_name", "unet")
    in_channels = int(model_cfg.pop("in_channels", 3))
    out_channels = int(model_cfg.pop("out_channels", 1))
    model = get_model(model_name, in_channels=in_channels, out_channels=out_channels, **model_cfg).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = get_loss(training.get("loss_name", "bce_dice"))
    result = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config)
    save_sample_predictions(model, val_loader, device, sanity_dir / "quick_train_predictions", max_samples=4)
    write_report(sanity_dir / "quick_train_report.md", config, device, result)
    print(f"Saved quick train report to: {sanity_dir / 'quick_train_report.md'}")


if __name__ == "__main__":
    main()
