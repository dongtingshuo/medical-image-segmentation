import pytest
import torch

from src.model_attention_unet import AttentionUNet
from src.model_factory import get_model
from src.model_unet import UNet


def test_unet_output_shape():
    model = UNet(in_channels=3, out_channels=1, base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_attention_unet_output_shape():
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_third_party_model_factory_skip_if_missing():
    pytest.importorskip("segmentation_models_pytorch")
    model = get_model(
        "unet_plus_plus",
        in_channels=3,
        out_channels=1,
        encoder_name="resnet18",
        encoder_weights=None,
    )
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)

