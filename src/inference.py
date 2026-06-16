import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model_factory import get_model
from src.utils import get_device, load_checkpoint
from src.visualization import make_overlay, save_prediction_result


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model_from_config(config, model_name_override=None):
    model_cfg = dict(config.get("model", {}))
    if model_name_override:
        model_cfg["model_name"] = model_name_override
    model_name = model_cfg.pop("model_name", "unet")
    in_channels = int(model_cfg.pop("in_channels", 3))
    out_channels = int(model_cfg.pop("out_channels", 1))
    return get_model(model_name, in_channels=in_channels, out_channels=out_channels, **model_cfg)


def preprocess_image(image_rgb, image_size):
    if isinstance(image_size, (list, tuple)):
        height, width = int(image_size[0]), int(image_size[1])
    else:
        height = width = int(image_size)
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().unsqueeze(0)
    return tensor, resized


def predict_array(image_rgb, config, checkpoint_path, threshold=0.5, device="auto", model_name_override=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    selected_device = get_device(device or config.get("device", "auto"))
    model = build_model_from_config(config, model_name_override=model_name_override).to(selected_device)
    load_checkpoint(checkpoint_path, model, selected_device)
    model.eval()

    image_size = config.get("data", {}).get("image_size", config.get("image_size", 256))
    tensor, resized_rgb = preprocess_image(image_rgb, image_size)
    start = time.time()
    with torch.no_grad():
        logits = model(tensor.to(selected_device))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    inference_time = time.time() - start
    pred_mask = (prob >= threshold).astype(np.float32)
    pred_mask_u8 = (pred_mask * 255).astype(np.uint8)
    overlay = make_overlay(resized_rgb, pred_mask_u8)
    lesion_ratio = float(pred_mask.mean())
    return {
        "image": resized_rgb,
        "mask": pred_mask_u8,
        "overlay": overlay,
        "lesion_ratio": lesion_ratio,
        "inference_time": inference_time,
        "device": str(selected_device),
    }


def predict_file(image_path, config, checkpoint_path, output_dir, threshold=0.5, device="auto", model_name_override=None):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = predict_array(
        image_rgb,
        config,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device=device,
        model_name_override=model_name_override,
    )
    paths = save_prediction_result(result["image"], result["mask"], output_dir, prefix=image_path.stem)
    result["paths"] = paths
    return result

