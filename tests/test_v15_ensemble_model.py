import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from src.ensemble_v15 import average_probability_files, greedy_select_members, macro_metrics
from src.model_segformer import SegFormerBinary


def test_streaming_average_matches_in_memory(tmp_path):
    arrays = [np.random.default_rng(index).random((3, 1, 8, 8), dtype=np.float32) for index in range(3)]
    paths = []
    for index, array in enumerate(arrays):
        path = tmp_path / f"member_{index}.npy"
        np.save(path, array.astype(np.float16))
        paths.append(path)
    actual = average_probability_files(paths)
    expected = np.mean([array.astype(np.float16).astype(np.float32) for array in arrays], axis=0)
    assert np.allclose(actual, expected, atol=1e-6)


def test_greedy_selection_keeps_best_member_when_extra_model_hurts(tmp_path):
    targets = np.zeros((2, 1, 8, 8), dtype=np.float32)
    targets[:, :, 2:6, 2:6] = 1.0
    good = targets * 0.98 + (1.0 - targets) * 0.02
    bad = 1.0 - good
    good_path, bad_path = tmp_path / "good.npy", tmp_path / "bad.npy"
    np.save(good_path, good.astype(np.float16))
    np.save(bad_path, bad.astype(np.float16))
    selected, probabilities, history = greedy_select_members(
        {"good": good_path, "bad": bad_path}, targets, min_improvement=0.0005, max_members=5
    )
    assert selected == ["good"]
    assert history[0]["dice"] == 1.0
    assert macro_metrics(probabilities, targets)["dice"] == 1.0


def test_segformer_wrapper_outputs_one_channel_same_size(monkeypatch):
    class FakeConfig:
        def __init__(self, **kwargs):
            self.num_labels = kwargs.get("num_labels", 1)

        @classmethod
        def from_pretrained(cls, _name, num_labels=1):
            config = cls()
            config.num_labels = num_labels
            return config

    class FakeSegformer(nn.Module):
        def __init__(self, _config=None):
            super().__init__()
            self.segformer = nn.Conv2d(3, 4, 1)
            self.decode = nn.Conv2d(4, 1, 1)

        @classmethod
        def from_pretrained(cls, _name, **_kwargs):
            return cls()

        def forward(self, pixel_values):
            features = self.segformer(pixel_values)
            logits = self.decode(torch.nn.functional.avg_pool2d(features, 4))
            return SimpleNamespace(logits=logits)

    fake_module = SimpleNamespace(SegformerConfig=FakeConfig, SegformerForSemanticSegmentation=FakeSegformer)
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    model = SegFormerBinary(encoder_name="nvidia/mit-b2", encoder_weights=None)
    output = model(torch.randn(2, 3, 32, 40))
    assert output.shape == (2, 1, 32, 40)
    assert model.encoder is model.model.segformer
