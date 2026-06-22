from pathlib import Path

import yaml

from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from scripts.run_error_analysis import run_error_analysis
from scripts.run_segmentation_comparison import run_comparison
from scripts.run_visualization_demo import run_visualization_demo


REQUIRED_KEYWORDS = [
    "Medical Disclaimer",
    "医学免责声明",
    "Current Limitations",
    "当前限制",
]


def _assert_keywords(path, extra_keywords=None):
    text = path.read_text(encoding="utf-8")
    for keyword in REQUIRED_KEYWORDS + list(extra_keywords or []):
        assert keyword in text, f"{keyword} missing in {path}"


def test_static_bilingual_documents_have_required_headings():
    _assert_keywords(Path("README.md"), ["Quick Demo", "快速演示"])
    _assert_keywords(Path("docs/EXPERIMENT_REPORT.md"))
    _assert_keywords(Path("examples/toy_segmentation_demo/README.md"))
    _assert_keywords(Path("MODEL_CARD.md"))
    dataset_text = Path("DATASET.md").read_text(encoding="utf-8")
    assert "ISIC 2017" in dataset_text
    assert "数据集说明" in dataset_text


def test_generated_reports_have_bilingual_headings(tmp_path):
    data_dir = tmp_path / "toy"
    create_toy_segmentation_data(data_dir, num_samples=4, image_size=64)
    comparison_config = {
        "seed": 11,
        "data": {
            "image_dir": str(data_dir / "images"),
            "mask_dir": str(data_dir / "masks"),
            "image_size": 64,
        },
        "training": {"batch_size": 2, "epochs": 1, "lr": 1e-3, "max_batches": 1},
        "models": [{"name": "unet", "base_channels": 4}],
        "losses": [{"name": "bce"}],
        "paths": {"output_dir": str(tmp_path / "comparison")},
    }
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(yaml.safe_dump(comparison_config), encoding="utf-8")
    comparison_dir = run_comparison(config_path)
    visualization_dir, visualization_report = run_visualization_demo(data_dir=data_dir, output_dir=tmp_path / "vis")
    analysis_dir, _, analysis_report = run_error_analysis(data_dir=data_dir, output_dir=tmp_path / "analysis")
    assert visualization_dir.exists()
    assert analysis_dir.exists()
    _assert_keywords(comparison_dir / "comparison_report.md")
    _assert_keywords(visualization_report)
    _assert_keywords(analysis_report)
