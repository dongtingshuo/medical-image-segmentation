import torch

from src.ema import ExponentialMovingAverage
from src.losses import build_loss
from src.model_unet import UNet


def test_hybrid_and_distillation_losses_backpropagate():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, logits.shape).float()
    soft_targets = torch.rand_like(targets)
    hybrid = build_loss({"loss": {"name": "hybrid_boundary"}})
    hybrid(logits, targets).backward(retain_graph=True)
    assert torch.isfinite(logits.grad).all()
    logits.grad.zero_()
    distillation = build_loss({"loss": {"name": "distillation", "temperature": 2.0}})
    distillation(logits, targets, soft_targets).backward()
    assert torch.isfinite(logits.grad).all()


def test_ema_round_trip_and_state_resume():
    model = UNet(in_channels=3, out_channels=1, base_channels=4)
    ema = ExponentialMovingAverage(model, decay=0.9)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    state = ema.state_dict()
    resumed = ExponentialMovingAverage(model, decay=0.5)
    resumed.load_state_dict(state)
    assert resumed.decay == 0.9
    resumed.store(model)
    resumed.copy_to(model)
    assert any(not torch.equal(model.state_dict()[name], original[name]) for name in original)
    resumed.restore(model)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
