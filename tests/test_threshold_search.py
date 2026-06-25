import torch

from src.threshold_search import best_threshold, parse_thresholds, summarize_threshold_predictions


def test_threshold_search_selects_best_threshold():
    probabilities = [torch.tensor([[[[0.2, 0.8], [0.6, 0.4]]]])]
    masks = [torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])]

    rows = summarize_threshold_predictions(probabilities, masks, thresholds=[0.3, 0.5, 0.7])
    best = best_threshold(rows, "dice")

    assert best["threshold"] == 0.5
    assert best["dice"] == 1.0
    assert best["iou"] == 1.0


def test_parse_thresholds_rejects_invalid_values():
    assert parse_thresholds("0.3,0.5") == [0.3, 0.5]
    try:
        parse_thresholds("1.2")
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("Expected invalid threshold to raise ValueError")
