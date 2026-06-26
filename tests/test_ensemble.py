from pathlib import Path

import pytest

from scripts.evaluate_ensemble import parse_member


def test_parse_member_uses_config_checkpoint_format():
    config, checkpoint = parse_member("config.yaml:best_model.pth")

    assert config == Path("config.yaml")
    assert checkpoint == Path("best_model.pth")


def test_parse_member_rejects_missing_separator():
    with pytest.raises(ValueError, match="CONFIG:CHECKPOINT"):
        parse_member("config.yaml")
