import numpy as np
import pytest

from src.inference import _load_model_cached, clear_model_cache, predict_array
from src.model_unet import UNet
from src.utils import save_checkpoint


def _runtime_config():
    return {
        "device": "cpu",
        "data": {"image_size": 32},
        "model": {"model_name": "unet", "in_channels": 3, "out_channels": 1, "base_channels": 4},
    }


def test_predict_array_uses_checkpoint_metadata_and_model_cache(tmp_path):
    config = _runtime_config()
    model = UNet(in_channels=3, out_channels=1, base_channels=4)
    checkpoint = tmp_path / "model.pth"
    save_checkpoint({"model_state_dict": model.state_dict(), "config": config, "epoch": 7}, checkpoint)
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    clear_model_cache()
    first = predict_array(image, config, checkpoint, device="cpu")
    second = predict_array(image, config, checkpoint, device="cpu")

    assert first["model_name"] == "unet"
    assert first["checkpoint_epoch"] == 7
    assert first["mask"].shape == (32, 32)
    assert second["mask"].shape == (32, 32)
    assert _load_model_cached.cache_info().hits >= 1


def test_predict_array_rejects_manual_model_mismatch(tmp_path):
    config = _runtime_config()
    model = UNet(in_channels=3, out_channels=1, base_channels=4)
    checkpoint = tmp_path / "model.pth"
    save_checkpoint({"model_state_dict": model.state_dict(), "config": config}, checkpoint)

    with pytest.raises(ValueError, match="does not match checkpoint model"):
        predict_array(
            np.zeros((32, 32, 3), dtype=np.uint8),
            config,
            checkpoint,
            device="cpu",
            model_name_override="attention_unet",
        )
