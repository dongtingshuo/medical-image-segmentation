import pytest

from src.benchmarking import benchmark_checkpoint, percentile, write_benchmark_report
from src.model_unet import UNet
from src.utils import save_checkpoint


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)


def test_cpu_benchmark_writes_reports(tmp_path):
    config = {
        "data": {"image_size": 32},
        "model": {"model_name": "unet", "in_channels": 3, "out_channels": 1, "base_channels": 4},
    }
    model = UNet(in_channels=3, out_channels=1, base_channels=4)
    checkpoint = tmp_path / "model.pth"
    save_checkpoint({"model_state_dict": model.state_dict(), "config": config}, checkpoint)

    result = benchmark_checkpoint(config, checkpoint, "cpu", warmup=0, iterations=1)
    assert result["parameters_total"] == sum(parameter.numel() for parameter in model.parameters())
    assert result["latency_mean_ms"] > 0
    assert result["cuda_peak_allocated_mb"] is None

    report = write_benchmark_report(tmp_path / "benchmark", [result])
    assert report.exists()
    assert (report.parent / "benchmark.csv").exists()
    assert (report.parent / "benchmark.json").exists()
