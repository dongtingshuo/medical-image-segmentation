import torch

from src.losses import build_loss, get_loss


def test_losses_return_scalar_and_finite():
    logits = torch.randn(2, 1, 16, 16)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
    for name in ["bce", "dice", "bce_dice", "focal", "focal_dice"]:
        loss = get_loss(name)
        value = loss(logits, targets)
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_build_loss_from_config_returns_scalar():
    logits = torch.randn(2, 1, 16, 16)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
    for name in ["bce", "dice", "bce_dice", "focal"]:
        loss = build_loss({"loss": {"name": name, "alpha": 0.25, "gamma": 2.0}})
        value = loss(logits, targets)
        assert value.ndim == 0
        assert torch.isfinite(value)
