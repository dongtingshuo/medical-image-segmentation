from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from scripts.run_visualization_demo import run_visualization_demo
from src.visualization import plot_training_curves


def test_visualization_demo_generates_overlay_and_report(tmp_path):
    data_dir = tmp_path / "toy"
    create_toy_segmentation_data(data_dir, num_samples=2, image_size=64)
    output_dir, report_path = run_visualization_demo(data_dir=data_dir, output_dir=tmp_path / "visualizations")
    assert (output_dir / "sample_001_overlay.png").exists()
    assert (output_dir / "sample_001_comparison.png").exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Visualization Overview / 可视化概述" in text
    assert "Medical Disclaimer / 医学免责声明" in text


def test_training_curve_plot_accepts_legacy_history_without_new_metrics(tmp_path):
    history = {
        "epoch": [1, 2],
        "train_loss": [1.0, 0.8],
        "val_loss": [1.1, 0.9],
        "val_dice": [0.2, 0.4],
        "val_iou": [0.1, 0.3],
        "val_precision": [0.3, 0.5],
        "val_recall": [0.2, 0.4],
    }
    output = tmp_path / "curves.png"
    plot_training_curves(history, output)
    assert output.exists()
