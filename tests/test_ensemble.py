from pathlib import Path

import pytest
import torch

from scripts.evaluate_ensemble import evaluate_ensemble, parse_member


def test_parse_member_uses_config_checkpoint_format():
    config, checkpoint = parse_member("config.yaml:best_model.pth")

    assert config == Path("config.yaml")
    assert checkpoint == Path("best_model.pth")


def test_parse_member_rejects_missing_separator():
    with pytest.raises(ValueError, match="CONFIG:CHECKPOINT"):
        parse_member("config.yaml")


def test_ensemble_primary_dice_is_macro_and_micro_dice_is_explicit():
    masks = torch.zeros(2, 1, 4, 4)
    masks[0, 0, 0, 0] = 1.0
    masks[1, 0, :2, :] = 1.0
    logits = torch.full_like(masks, -10.0)
    logits[0, 0, 0, 0] = 10.0
    members = [{"model": torch.nn.Identity()}, {"model": torch.nn.Identity()}]

    metrics = evaluate_ensemble(members, [(logits, masks)], torch.device("cpu"), threshold=0.5)

    assert metrics["dice"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["micro_dice"] == pytest.approx(0.2, abs=1e-6)
