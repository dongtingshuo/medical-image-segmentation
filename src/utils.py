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


def cuda_runtime_is_usable():
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        x = torch.randn(1, 3, 16, 16, device="cuda")
        conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).cuda()
        y = conv(x)
        torch.cuda.synchronize()
        return bool(y.is_cuda and y.numel() > 0), ""
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def get_device(device="auto", require_cuda_runtime=False):
    requested = str(device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            ok, error = cuda_runtime_is_usable()
            if ok:
                return torch.device("cuda")
            message = (
                "CUDA is reported as available, but a runtime smoke test failed. "
                f"Error: {error}. On Kaggle, run scripts/kaggle_prepare_gpu.py before formal training."
            )
            if require_cuda_runtime:
                raise RuntimeError(message)
            print(message)
        if require_cuda_runtime:
            raise RuntimeError("GPU training is required, but CUDA is not available.")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            message = "CUDA was requested but is not available."
            if require_cuda_runtime:
                raise RuntimeError(message)
            print(f"{message} Falling back to CPU.")
            return torch.device("cpu")
        ok, error = cuda_runtime_is_usable()
        if not ok:
            message = (
                "CUDA was requested, but a runtime smoke test failed. "
                f"Error: {error}. On Kaggle, run scripts/kaggle_prepare_gpu.py before formal training."
            )
            if require_cuda_runtime:
                raise RuntimeError(message)
            print(f"{message} Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
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
