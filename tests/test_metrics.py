import torch

from src.metrics import boundary_f1_score, dice_score, iou_score, specificity_score


def test_dice_and_iou_perfect_prediction():
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    assert torch.isclose(dice_score(logits, targets), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(iou_score(logits, targets), torch.tensor(1.0), atol=1e-5)


def test_dice_and_iou_batch_safe_for_empty_masks():
    targets = torch.zeros(2, 1, 4, 4)
    logits = torch.full((2, 1, 4, 4), -10.0)
    assert torch.isfinite(dice_score(logits, targets))
    assert torch.isfinite(iou_score(logits, targets))


def test_specificity_and_boundary_f1_for_perfect_prediction():
    targets = torch.zeros(1, 1, 16, 16)
    targets[:, :, 4:12, 4:12] = 1.0
    logits = torch.where(targets > 0, torch.tensor(10.0), torch.tensor(-10.0))
    assert torch.isclose(specificity_score(logits, targets), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(boundary_f1_score(logits, targets), torch.tensor(1.0), atol=1e-5)


def test_boundary_f1_detects_large_boundary_shift():
    targets = torch.zeros(1, 1, 32, 32)
    targets[:, :, 4:12, 4:12] = 1.0
    shifted = torch.zeros_like(targets)
    shifted[:, :, 20:28, 20:28] = 1.0
    logits = torch.where(shifted > 0, torch.tensor(10.0), torch.tensor(-10.0))
    assert boundary_f1_score(logits, targets, tolerance=1) < 0.1
