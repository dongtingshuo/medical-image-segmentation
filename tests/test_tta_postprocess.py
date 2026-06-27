import pytest
import torch

from scripts.evaluate_tta_postprocess import evaluate_tta_postprocess, parse_scales


def test_parse_scales_accepts_comma_separated_values():
    assert parse_scales("0.875, 1.0,1.125") == [0.875, 1.0, 1.125]


def test_parse_scales_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        parse_scales("1.0,0")


def test_tta_primary_dice_is_macro_and_micro_dice_is_explicit():
    masks = torch.zeros(2, 1, 4, 4)
    masks[0, 0, 0, 0] = 1.0
    masks[1, 0, :2, :] = 1.0
    logits = torch.full_like(masks, -10.0)
    logits[0, 0, 0, 0] = 10.0

    metrics = evaluate_tta_postprocess(
        torch.nn.Identity(),
        [(logits, masks)],
        torch.device("cpu"),
        threshold=0.5,
    )

    assert metrics["dice"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["micro_dice"] == pytest.approx(0.2, abs=1e-6)
