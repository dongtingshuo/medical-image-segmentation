import argparse
import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import SkinLesionDataset, get_val_transforms
from src.losses import get_loss
from src.metrics import dice_score, iou_score
from src.model_factory import get_model
from src.utils import data_path, get_device, load_config, set_seed
from src.visualization import save_sample_predictions


def build_model(config, device):
    model_cfg = copy.deepcopy(config.get("model", {}))
    model_name = model_cfg.pop("model_name", "unet")
    in_channels = int(model_cfg.pop("in_channels", 3))
    out_channels = int(model_cfg.pop("out_channels", 1))
    return get_model(model_name, in_channels=in_channels, out_channels=out_channels, **model_cfg).to(device)


def build_optimizer(model, config):
    training = config.get("training", {})
    lr = float(training.get("lr", 1e-4))
    name = str(training.get("optimizer", "adam")).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr)


def save_curves(history, output_path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["loss"])
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["dice"])
    axes[1].set_title("Dice")
    axes[2].plot(epochs, history["iou"])
    axes[2].set_title("IoU")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(path, config, device, sample_count, history, image_dir):
    loss_drop = history["loss"][0] - history["loss"][-1]
    dice_gain = history["dice"][-1] - history["dice"][0]
    iou_gain = history["iou"][-1] - history["iou"][0]
    has_nan = any(math.isnan(v) for key in ["loss", "dice", "iou"] for v in history[key])
    lines = [
        "# Small Batch Overfit Report",
        "",
        f"- Model: {config.get('model', {}).get('model_name', 'unet')}",
        f"- Device: {device}",
        f"- Samples: {sample_count}",
        f"- Epochs: {len(history['epoch'])}",
        f"- Initial loss: {history['loss'][0]:.6f}",
        f"- Final loss: {history['loss'][-1]:.6f}",
        f"- Loss drop: {loss_drop:.6f}",
        f"- Initial Dice: {history['dice'][0]:.6f}",
        f"- Final Dice: {history['dice'][-1]:.6f}",
        f"- Dice gain: {dice_gain:.6f}",
        f"- Final IoU: {history['iou'][-1]:.6f}",
        f"- IoU gain: {iou_gain:.6f}",
        f"- NaN detected: {has_nan}",
        f"- Prediction samples: {image_dir}",
        "",
        "## Interpretation",
        "",
        "- Normal: loss clearly decreases and Dice/IoU increase.",
        "- If loss does not decrease, check mask alignment, learning rate, loss function, and model output logits.",
        "- If predictions remain all black/all white, do not start formal long training.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / "sanity_check"
    pred_dir = output_dir / "overfit_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset = SkinLesionDataset(data_path(config, "train_images_dir"), data_path(config, "train_masks_dir"), get_val_transforms(config))
    sample_count = min(args.sample_count, len(dataset))
    subset = Subset(dataset, list(range(sample_count)))
    loader = DataLoader(subset, batch_size=sample_count, shuffle=False, num_workers=0)

    model = build_model(config, device)
    criterion = get_loss(config.get("training", {}).get("loss_name", "bce_dice"))
    optimizer = build_optimizer(model, config)
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)

    history = {"epoch": [], "loss": [], "dice": [], "iou": []}
    use_amp = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(images)
            dice = dice_score(logits, masks).item()
            iou = iou_score(logits, masks).item()
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["dice"].append(dice)
        history["iou"].append(iou)
        print(f"epoch={epoch:03d} loss={loss.item():.6f} dice={dice:.6f} iou={iou:.6f}")

    save_curves(history, output_dir / "overfit_curves.png")
    save_sample_predictions(model, loader, device, pred_dir, max_samples=min(8, sample_count))
    write_report(output_dir / "overfit_report.md", config, device, sample_count, history, pred_dir)
    print(f"Saved overfit report to: {output_dir / 'overfit_report.md'}")


if __name__ == "__main__":
    main()
