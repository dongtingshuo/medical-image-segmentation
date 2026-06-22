import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model_factory import get_model
from src.utils import canonical_model_name, checkpoint_model_config, get_device, load_checkpoint, load_checkpoint_payload
from src.visualization import make_overlay, save_prediction_result


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _effective_model_config(config, checkpoint=None, model_name_override=None):
    runtime_model_cfg = dict(config.get("model", {}))
    saved_model_cfg = checkpoint_model_config(checkpoint or {})
    model_cfg = saved_model_cfg or runtime_model_cfg
    saved_name = saved_model_cfg.get("model_name", saved_model_cfg.get("name"))
    if model_name_override and saved_name and canonical_model_name(model_name_override) != canonical_model_name(saved_name):
        raise ValueError(
            f"Selected model `{model_name_override}` does not match checkpoint model `{saved_name}`. "
            "Choose Auto or the checkpoint's model architecture."
        )
    if model_name_override:
        model_cfg["model_name"] = model_name_override
    if saved_model_cfg:
        # The checkpoint already contains encoder parameters; do not download ImageNet weights again.
        model_cfg["encoder_weights"] = None
    return model_cfg


def build_model_from_config(config, model_name_override=None, checkpoint=None):
    model_cfg = _effective_model_config(config, checkpoint=checkpoint, model_name_override=model_name_override)
    model_name = model_cfg.pop("model_name", model_cfg.pop("name", "unet"))
    if "encoder_name" not in model_cfg and model_cfg.get("encoder"):
        model_cfg["encoder_name"] = model_cfg["encoder"]
    model_cfg.pop("encoder", None)
    in_channels = int(model_cfg.pop("in_channels", 3))
    out_channels = int(model_cfg.pop("out_channels", 1))
    return get_model(model_name, in_channels=in_channels, out_channels=out_channels, **model_cfg)


@lru_cache(maxsize=2)
def _load_model_cached(checkpoint_path, checkpoint_mtime_ns, device_name, config_json, model_name_override):
    del checkpoint_mtime_ns
    config = json.loads(config_json)
    device = torch.device(device_name)
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
    effective_model_cfg = _effective_model_config(
        config,
        checkpoint=checkpoint,
        model_name_override=model_name_override or None,
    )
    model = build_model_from_config(
        config,
        checkpoint=checkpoint,
        model_name_override=model_name_override or None,
    ).to(device)
    load_checkpoint(
        checkpoint_path,
        model,
        device,
        expected_model_config=effective_model_cfg,
        checkpoint=checkpoint,
    )
    model.eval()
    model_name = effective_model_cfg.get("model_name", effective_model_cfg.get("name", "unet"))
    return model, str(model_name), checkpoint.get("epoch")


def clear_model_cache():
    _load_model_cached.cache_clear()


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
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    selected_device = get_device(device or config.get("device", "auto"))
    config_json = json.dumps(config, ensure_ascii=True, sort_keys=True, default=str)
    model, model_name, checkpoint_epoch = _load_model_cached(
        str(checkpoint_path),
        checkpoint_path.stat().st_mtime_ns,
        str(selected_device),
        config_json,
        model_name_override or "",
    )

    image_size = config.get("data", {}).get("image_size", config.get("image_size", 256))
    tensor, resized_rgb = preprocess_image(image_rgb, image_size)
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    start = time.time()
    with torch.no_grad():
        logits = model(tensor.to(selected_device))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
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
        "model_name": model_name,
        "checkpoint_epoch": checkpoint_epoch,
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
