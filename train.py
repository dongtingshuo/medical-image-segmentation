import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import SkinLesionDataset, get_train_transforms, get_val_transforms
from src.losses import build_loss
from src.model_factory import get_model
from src.trainer import train_model
from src.utils import (
    create_dirs,
    data_path,
    get_device,
    load_config,
    make_torch_generator,
    seed_worker,
    set_seed,
)


def _require_dir(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}. Please update the YAML config.")
    return path


def build_optimizer(model, config):
    training = config.get("training", {})
    lr = float(training.get("lr", 1e-4))
    name = str(training.get("optimizer", "adam")).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, config):
    training = config.get("training", {})
    name = str(training.get("scheduler", "none")).lower()
    epochs = int(training.get("epochs", 1))
    if name in {"none", "null"}:
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    raise ValueError(f"Unsupported scheduler: {name}")


def build_weighted_sampler(dataset, config):
    data_cfg = config.get("data", {})
    weights_path = data_cfg.get("sample_weights_csv")
    if not weights_path:
        return None
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"sample_weights_csv does not exist: {weights_path}")
    stem_column = data_cfg.get("sample_weight_stem_column", "stem")
    weight_column = data_cfg.get("sample_weight_column", "weight")
    weights_by_stem = {}
    with weights_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if stem_column not in (reader.fieldnames or []) or weight_column not in (reader.fieldnames or []):
            raise ValueError(
                f"sample_weights_csv must contain `{stem_column}` and `{weight_column}` columns: {weights_path}"
            )
        for row in reader:
            weights_by_stem[str(row[stem_column])] = float(row[weight_column])
    weights = []
    for image_path, _ in dataset.pairs:
        weight = float(weights_by_stem.get(image_path.stem, 1.0))
        if weight <= 0:
            raise ValueError(f"Sample weight must be positive for `{image_path.stem}`, got {weight}")
        weights.append(weight)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=make_torch_generator(int(config.get("seed", 42))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--resume", default=None, help="Resume training from a project checkpoint, usually last_model.pth.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("reproducibility", {}).get("deterministic", True))
    set_seed(seed, deterministic=deterministic)
    device = get_device(config.get("device", "auto"), require_cuda_runtime=bool(config.get("require_gpu", False)))
    paths = config.get("paths", {})
    create_dirs(paths.get("output_dir", "outputs"), paths.get("checkpoint_dir", "checkpoints"))

    train_images = _require_dir(data_path(config, "train_images_dir"), "train_images_dir")
    train_masks = _require_dir(data_path(config, "train_masks_dir"), "train_masks_dir")
    val_images = _require_dir(data_path(config, "val_images_dir"), "val_images_dir")
    val_masks = _require_dir(data_path(config, "val_masks_dir"), "val_masks_dir")

    train_dataset = SkinLesionDataset(train_images, train_masks, transform=get_train_transforms(config))
    val_dataset = SkinLesionDataset(val_images, val_masks, transform=get_val_transforms(config))

    training = config.get("training", {})
    train_sampler = build_weighted_sampler(train_dataset, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 8)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(training.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 1),
    )

    model_cfg = dict(config.get("model", {}))
    model_name = model_cfg.pop("model_name", model_cfg.pop("name", "unet"))
    if "encoder_name" not in model_cfg and model_cfg.get("encoder"):
        model_cfg["encoder_name"] = model_cfg["encoder"]
    model_cfg.pop("encoder", None)
    in_channels = int(model_cfg.pop("in_channels", 3))
    out_channels = int(model_cfg.pop("out_channels", 1))
    model = get_model(model_name, in_channels=in_channels, out_channels=out_channels, **model_cfg).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_loss(config)

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Deterministic mode: {deterministic}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    resume_path = args.resume or training.get("resume_from")
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, resume_path=resume_path)


if __name__ == "__main__":
    main()
