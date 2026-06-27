from pathlib import Path

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_to_rgb(image_tensor):
    image = image_tensor.detach().cpu().float().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)
    return image


def mask_to_uint8(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()
    return (mask > 0.5).astype(np.uint8) * 255


def make_overlay(image_rgb, mask, color=(255, 0, 0), alpha=0.45):
    image = image_rgb.copy()
    mask_bool = np.asarray(mask).squeeze() > 0
    overlay = image.copy()
    overlay[mask_bool] = color
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def make_false_positive_map(pred_mask, true_mask):
    pred = np.asarray(pred_mask).squeeze() > 0
    true = np.asarray(true_mask).squeeze() > 0
    return ((pred & ~true).astype(np.uint8) * 255)


def make_false_negative_map(pred_mask, true_mask):
    pred = np.asarray(pred_mask).squeeze() > 0
    true = np.asarray(true_mask).squeeze() > 0
    return ((~pred & true).astype(np.uint8) * 255)


def make_side_by_side_comparison(image_rgb, true_mask, pred_mask):
    true_u8 = mask_to_uint8(true_mask)
    pred_u8 = mask_to_uint8(pred_mask)
    fp_u8 = make_false_positive_map(pred_u8, true_u8)
    fn_u8 = make_false_negative_map(pred_u8, true_u8)
    true_overlay = make_overlay(image_rgb, true_u8, color=(0, 200, 0), alpha=0.45)
    pred_overlay = make_overlay(image_rgb, pred_u8, color=(255, 0, 0), alpha=0.45)
    fp_rgb = np.stack([fp_u8, np.zeros_like(fp_u8), np.zeros_like(fp_u8)], axis=-1)
    fn_rgb = np.stack([np.zeros_like(fn_u8), np.zeros_like(fn_u8), fn_u8], axis=-1)
    return np.concatenate([image_rgb, true_overlay, pred_overlay, fp_rgb, fn_rgb], axis=1)


def save_visualization_set(image_rgb, true_mask, pred_mask, output_dir, prefix="sample_001"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_u8 = mask_to_uint8(true_mask)
    pred_u8 = mask_to_uint8(pred_mask)
    overlay = make_overlay(image_rgb, pred_u8)
    fp_map = make_false_positive_map(pred_u8, true_u8)
    fn_map = make_false_negative_map(pred_u8, true_u8)
    comparison = make_side_by_side_comparison(image_rgb, true_u8, pred_u8)
    paths = {
        "overlay": output_dir / f"{prefix}_overlay.png",
        "false_positive": output_dir / f"{prefix}_false_positive.png",
        "false_negative": output_dir / f"{prefix}_false_negative.png",
        "comparison": output_dir / f"{prefix}_comparison.png",
    }
    cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(paths["false_positive"]), fp_map)
    cv2.imwrite(str(paths["false_negative"]), fn_map)
    cv2.imwrite(str(paths["comparison"]), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    return paths


def save_prediction_result(image_rgb, pred_mask, output_dir, prefix="prediction", true_mask=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_u8 = mask_to_uint8(pred_mask)
    overlay = make_overlay(image_rgb, pred_mask_u8)
    cv2.imwrite(str(output_dir / f"{prefix}_image.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_pred_mask.png"), pred_mask_u8)
    cv2.imwrite(str(output_dir / f"{prefix}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if true_mask is not None:
        cv2.imwrite(str(output_dir / f"{prefix}_true_mask.png"), mask_to_uint8(true_mask))
    lesion_ratio = float((pred_mask_u8 > 0).mean())
    (output_dir / f"{prefix}_lesion_ratio.txt").write_text(f"{lesion_ratio:.6f}\n", encoding="utf-8")
    return {
        "image": output_dir / f"{prefix}_image.png",
        "pred_mask": output_dir / f"{prefix}_pred_mask.png",
        "overlay": output_dir / f"{prefix}_overlay.png",
        "lesion_ratio": lesion_ratio,
    }


def plot_training_curves(history, output_path):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0, 0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, history.get("val_dice", []), label="Dice")
    axes[0, 1].plot(epochs, history.get("val_iou", []), label="IoU")
    axes[0, 1].set_title("Dice / IoU")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, history.get("val_precision", []), label="Precision")
    axes[1, 0].plot(epochs, history.get("val_recall", []), label="Recall / Sensitivity")
    if history.get("val_specificity"):
        axes[1, 0].plot(epochs, history["val_specificity"], label="Specificity")
    axes[1, 0].set_title("Pixel Classification")
    axes[1, 0].legend()
    if history.get("val_boundary_f1"):
        axes[1, 1].plot(epochs, history["val_boundary_f1"], label="Boundary F1")
    axes[1, 1].set_title("Boundary Quality")
    if axes[1, 1].lines:
        axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_sample_predictions(model, dataloader, device, output_dir, threshold=0.5, max_samples=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for images, masks in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        for i in range(images.size(0)):
            image_rgb = tensor_to_rgb(images[i])
            pred_mask = (probs[i, 0] >= threshold).astype(np.float32)
            true_mask = masks[i, 0].cpu().numpy()
            save_prediction_result(image_rgb, pred_mask, output_dir, prefix=f"sample_{saved:03d}", true_mask=true_mask)
            saved += 1
            if saved >= max_samples:
                return
