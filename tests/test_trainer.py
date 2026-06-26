import csv

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.losses import get_loss
from src.model_unet import UNet
from src.trainer import train_model
from src.utils import CHECKPOINT_FORMAT_VERSION, load_checkpoint_payload, set_seed


def test_train_model_writes_versioned_best_checkpoint_and_unambiguous_metrics(tmp_path):
    set_seed(5, deterministic=True)
    images = torch.rand(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    masks[:, :, 8:24, 8:24] = 1.0
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
    model = UNet(in_channels=3, out_channels=1, base_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {
        "experiment_name": "trainer_test",
        "mixed_precision": False,
        "data": {"image_size": 32},
        "model": {"model_name": "unet", "in_channels": 3, "out_channels": 1, "base_channels": 4},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 1e-3,
            "loss_name": "bce",
            "early_stopping": {"enabled": False, "monitor": "val_dice", "mode": "max"},
        },
        "augmentation": {"enabled": False},
        "paths": {
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }

    result = train_model(
        model,
        loader,
        loader,
        get_loss("bce"),
        optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        config=config,
    )

    checkpoint = load_checkpoint_payload(result["best_checkpoint"])
    assert checkpoint["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["metadata"]["monitor"] == "val_dice"

    with (tmp_path / "outputs" / "experiment_results.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["best_epoch"] == "1"
    assert row["best_val_loss"] == row["val_loss_at_best_epoch"]


def test_train_model_resumes_from_last_checkpoint(tmp_path):
    set_seed(7, deterministic=True)
    images = torch.rand(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    masks[:, :, 10:22, 10:22] = 1.0
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
    config = {
        "experiment_name": "resume_test",
        "mixed_precision": False,
        "data": {"image_size": 32},
        "model": {"model_name": "unet", "in_channels": 3, "out_channels": 1, "base_channels": 4},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 1e-3,
            "loss_name": "bce",
            "early_stopping": {"enabled": False, "monitor": "val_dice", "mode": "max"},
        },
        "augmentation": {"enabled": False},
        "paths": {
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }
    model = UNet(in_channels=3, out_channels=1, base_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = train_model(
        model,
        loader,
        loader,
        get_loss("bce"),
        optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        config=config,
    )

    resumed_config = {**config, "training": {**config["training"], "epochs": 2}}
    resumed_model = UNet(in_channels=3, out_channels=1, base_channels=4)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    second = train_model(
        resumed_model,
        loader,
        loader,
        get_loss("bce"),
        resumed_optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        config=resumed_config,
        resume_path=first["last_checkpoint"],
    )

    assert second["history"]["epoch"] == [1, 2]
    last_checkpoint = load_checkpoint_payload(second["last_checkpoint"])
    assert last_checkpoint["epoch"] == 2
    assert last_checkpoint["history"]["epoch"] == [1, 2]
    assert "optimizer_state_dict" in last_checkpoint
