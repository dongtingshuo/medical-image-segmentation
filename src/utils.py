import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml


CHECKPOINT_FORMAT_VERSION = 2
MODEL_NAME_ALIASES = {"unetplusplus": "unet_plus_plus"}


def canonical_model_name(model_name):
    normalized = str(model_name).lower().replace("-", "_")
    return MODEL_NAME_ALIASES.get(normalized, normalized)


def set_seed(seed=42, deterministic=True):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    transform = getattr(getattr(worker_info, "dataset", None), "transform", None)
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(worker_seed)


def make_torch_generator(seed=42):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


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


def load_checkpoint_payload(path, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This project requires a PyTorch version that supports safe checkpoint loading with "
            "`weights_only=True`. Install the pinned dependencies from requirements.txt."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint could not be loaded safely: {path}. Only use checkpoints produced by this project "
            "or downloaded from its verified GitHub Release."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}: expected a dictionary.")
    return checkpoint


def checkpoint_model_config(checkpoint):
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return {}
    model_config = config.get("model", {})
    return dict(model_config) if isinstance(model_config, dict) else {}


def model_architecture_signature(model_config):
    model_config = dict(model_config or {})
    model_name = model_config.get("model_name", model_config.get("name", "unet"))
    encoder_name = model_config.get("encoder_name", model_config.get("encoder"))
    signature = {
        "model_name": canonical_model_name(model_name),
        "in_channels": int(model_config.get("in_channels", 3)),
        "out_channels": int(model_config.get("out_channels", 1)),
    }
    if signature["model_name"] in {"unet", "attention_unet"}:
        signature["base_channels"] = int(model_config.get("base_channels", 32))
    elif encoder_name:
        signature["encoder_name"] = str(encoder_name).lower()
    return signature


def validate_checkpoint_model_config(checkpoint, expected_model_config):
    saved_model_config = checkpoint_model_config(checkpoint)
    if not saved_model_config or not expected_model_config:
        return
    saved_signature = model_architecture_signature(saved_model_config)
    expected_signature = model_architecture_signature(expected_model_config)
    if saved_signature != expected_signature:
        raise ValueError(
            "Checkpoint architecture does not match the selected model configuration. "
            f"checkpoint={saved_signature}, selected={expected_signature}. "
            "Use the checkpoint's embedded configuration or the matching YAML file."
        )


def load_checkpoint(path, model, device, optimizer=None, expected_model_config=None, checkpoint=None):
    checkpoint = checkpoint or load_checkpoint_payload(path, device=device)
    validate_checkpoint_model_config(checkpoint, expected_model_config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint parameters do not match the constructed model. Check model_name, encoder_name, "
            "in_channels, out_channels, and base_channels."
        ) from exc
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"Optimizer state was not restored: {exc}", RuntimeWarning, stacklevel=2)
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
