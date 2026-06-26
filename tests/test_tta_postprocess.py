import pytest

from scripts.evaluate_tta_postprocess import parse_scales


def test_parse_scales_accepts_comma_separated_values():
    assert parse_scales("0.875, 1.0,1.125") == [0.875, 1.0, 1.125]


def test_parse_scales_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        parse_scales("1.0,0")
