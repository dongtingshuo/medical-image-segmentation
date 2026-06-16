import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device="auto"):
    requested = str(device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    return torch.device(requested)


def create_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, device, optimizer=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:02d}m {seconds:02d}s"


def now():
    return time.time()


def path_from_config(config, key, default=None):
    return config.get("paths", {}).get(key, config.get(key, default))


def data_path(config, key):
    return config.get("data", {}).get(key, config.get(key))

