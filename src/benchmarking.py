import csv
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

try:
    import resource
except ImportError:  # Windows does not provide the resource module.
    resource = None

from src.inference import build_model_from_config
from src.utils import checkpoint_model_config, load_checkpoint, load_checkpoint_payload


def percentile(values, quantile):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sequence.")
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def process_peak_rss_mb():
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return float(peak) / divisor


def model_size_mb(model):
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return (parameter_bytes + buffer_bytes) / (1024**2)


def _device_label(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "CPU"


def benchmark_checkpoint(config, checkpoint_path, device_name, warmup=5, iterations=20, batch_size=1):
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative.")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but CUDA is not available.")

    checkpoint_path = Path(checkpoint_path)
    payload = load_checkpoint_payload(checkpoint_path, device="cpu")
    model = build_model_from_config(config, checkpoint=payload).to(device)
    expected_model_config = checkpoint_model_config(payload) or config.get("model", {})
    load_checkpoint(
        checkpoint_path,
        model,
        device,
        expected_model_config=expected_model_config,
        checkpoint=payload,
    )
    del payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    image_size = config.get("data", {}).get("image_size", 256)
    if isinstance(image_size, (list, tuple)):
        height, width = int(image_size[0]), int(image_size[1])
    else:
        height = width = int(image_size)
    sample = torch.randn(int(batch_size), 3, height, width, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        latencies_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    expected_shape = (int(batch_size), 1, height, width)
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}; expected {expected_shape}")

    result = {
        "device": str(device),
        "device_name": _device_label(device),
        "precision": "float32",
        "batch_size": int(batch_size),
        "image_height": height,
        "image_width": width,
        "warmup_iterations": int(warmup),
        "measured_iterations": int(iterations),
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "latency_std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "throughput_images_per_second": 1000.0 * int(batch_size) / statistics.fmean(latencies_ms),
        "process_peak_rss_mb": process_peak_rss_mb(),
        "cuda_peak_allocated_mb": None,
        "cuda_peak_reserved_mb": None,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_state_size_mb": model_size_mb(model),
        "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        result["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2)
    return result


def write_benchmark_report(output_dir, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("No benchmark results to write.")
    (output_dir / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Inference Benchmark",
        "",
        "Forward-pass timing uses FP32, includes device synchronization, and excludes model loading and preprocessing.",
        "",
        "| Device | Input | Mean latency (ms) | P95 (ms) | Throughput (img/s) | Peak memory (MB) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        peak_memory = row["cuda_peak_allocated_mb"]
        if peak_memory is None:
            peak_memory = row["process_peak_rss_mb"]
        peak_memory_text = f"{peak_memory:.2f}" if peak_memory is not None else "Not available"
        lines.append(
            f"| {row['device_name']} ({row['device']}) | {row['image_height']}x{row['image_width']} "
            f"(batch {row['batch_size']}) | {row['latency_mean_ms']:.3f} | {row['latency_p95_ms']:.3f} | "
            f"{row['throughput_images_per_second']:.3f} | {peak_memory_text} |"
        )
    first = rows[0]
    lines.extend(
        [
            "",
            f"- Parameters: `{first['parameters_total']:,}`",
            f"- Model state size: `{first['model_state_size_mb']:.2f} MB`",
            f"- Checkpoint size: `{first['checkpoint_size_mb']:.2f} MB`",
            "- CPU memory is process peak RSS; CUDA memory is peak allocated device memory during model residency and inference.",
            "",
        ]
    )
    (output_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir / "benchmark.md"
