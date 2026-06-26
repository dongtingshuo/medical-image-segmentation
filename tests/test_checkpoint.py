import pytest
import torch

from src.model_unet import UNet
from src.trainer import _time_stop_after_seconds
from src.utils import load_checkpoint, load_checkpoint_payload, save_checkpoint


def _config(base_channels=4):
    return {
        "model": {
            "model_name": "unet",
            "in_channels": 3,
            "out_channels": 1,
            "base_channels": base_channels,
        }
    }


def test_checkpoint_round_trip_uses_safe_dictionary_payload(tmp_path):
    source = UNet(in_channels=3, out_channels=1, base_channels=4)
    path = tmp_path / "model.pth"
    save_checkpoint({"model_state_dict": source.state_dict(), "config": _config(), "epoch": 3}, path)

    payload = load_checkpoint_payload(path)
    target = UNet(in_channels=3, out_channels=1, base_channels=4)
    load_checkpoint(path, target, torch.device("cpu"), expected_model_config=_config()["model"], checkpoint=payload)

    assert payload["epoch"] == 3
    for source_value, target_value in zip(
        source.state_dict().values(),
        target.state_dict().values(),
    ):
        assert torch.equal(source_value, target_value)


def test_checkpoint_rejects_architecture_mismatch(tmp_path):
    model = UNet(in_channels=3, out_channels=1, base_channels=4)
    path = tmp_path / "model.pth"
    save_checkpoint({"model_state_dict": model.state_dict(), "config": _config()}, path)
    payload = load_checkpoint_payload(path)

    with pytest.raises(ValueError, match="architecture does not match"):
        load_checkpoint(
            path,
            UNet(in_channels=3, out_channels=1, base_channels=8),
            torch.device("cpu"),
            expected_model_config=_config(base_channels=8)["model"],
            checkpoint=payload,
        )


def test_time_budget_uses_safe_stop_window():
    assert _time_stop_after_seconds({}) is None
    assert _time_stop_after_seconds({"max_runtime_minutes": 60, "safe_stop_minutes": 10}) == 3000.0
    assert _time_stop_after_seconds({"max_runtime_minutes": 5, "safe_stop_minutes": 10}) == 0.0
