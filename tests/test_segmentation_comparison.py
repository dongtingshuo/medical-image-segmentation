import json

import yaml

from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from scripts.run_segmentation_comparison import run_comparison


def test_run_segmentation_comparison_outputs_reports(tmp_path):
    data_dir = tmp_path / "toy"
    create_toy_segmentation_data(data_dir, num_samples=4, image_size=64)
    config = {
        "seed": 7,
        "data": {
            "image_dir": str(data_dir / "images"),
            "mask_dir": str(data_dir / "masks"),
            "image_size": 64,
        },
        "training": {"batch_size": 2, "epochs": 1, "lr": 1e-3, "max_batches": 1},
        "models": [{"name": "unet", "base_channels": 4}],
        "losses": [{"name": "bce"}, {"name": "dice"}],
        "paths": {"output_dir": str(tmp_path / "comparison")},
    }
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_dir = run_comparison(config_path)
    assert (output_dir / "comparison_results.json").exists()
    assert (output_dir / "comparison_results.csv").exists()
    report = output_dir / "comparison_report.md"
    assert report.exists()
    payload = json.loads((output_dir / "comparison_results.json").read_text(encoding="utf-8"))
    assert len(payload) == 2
    text = report.read_text(encoding="utf-8")
    assert "Experiment Objective / 实验目的" in text
    assert "Medical Disclaimer / 医学免责声明" in text
