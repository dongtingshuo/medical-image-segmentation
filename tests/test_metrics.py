import torch

from src.metrics import dice_score, iou_score


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

