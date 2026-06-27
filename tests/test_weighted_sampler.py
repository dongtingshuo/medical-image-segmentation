from pathlib import Path

from train import build_weighted_sampler


class DummyDataset:
    def __init__(self):
        self.pairs = [
            (Path("sample_a.jpg"), Path("sample_a.png")),
            (Path("sample_b.jpg"), Path("sample_b.png")),
        ]

    def __len__(self):
        return len(self.pairs)


def test_build_weighted_sampler_from_csv(tmp_path):
    weights_path = tmp_path / "weights.csv"
    weights_path.write_text("stem,weight\nsample_a,3.0\nsample_b,1.0\n", encoding="utf-8")

    sampler = build_weighted_sampler(DummyDataset(), {"data": {"sample_weights_csv": str(weights_path)}})

    assert sampler is not None
    assert sampler.num_samples == 2


def test_build_weighted_sampler_is_optional():
    assert build_weighted_sampler(DummyDataset(), {"data": {}}) is None


def test_source_sampling_combines_quality_proportions_and_hard_weights(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "stem,source,status\nsample_a,isic17,accepted\nsample_b,ph2,accepted\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.csv"
    weights.write_text("stem,weight\nsample_a,2.0\nsample_b,1.0\n", encoding="utf-8")
    config = {
        "data": {
            "sample_weights_csv": str(weights),
            "source_sampling": {
                "manifest": str(manifest),
                "proportions": {"isic17": 0.65, "ph2": 0.10},
            },
        }
    }
    sampler = build_weighted_sampler(DummyDataset(), config)
    assert sampler is not None
    assert sampler.weights[0] > sampler.weights[1]
