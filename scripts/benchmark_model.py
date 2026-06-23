import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.benchmarking import benchmark_checkpoint, write_benchmark_report  # noqa: E402
from src.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Benchmark checkpoint inference on CPU and/or CUDA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cpu-iterations", type=int, default=10)
    parser.add_argument("--cuda-iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/benchmark")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("Skipping CUDA benchmark because CUDA is not available.")
            continue
        iterations = args.cuda_iterations if device == "cuda" else args.cpu_iterations
        print(f"Benchmarking {device} for {iterations} measured iterations...", flush=True)
        rows.append(
            benchmark_checkpoint(
                config,
                args.checkpoint,
                device,
                warmup=args.warmup,
                iterations=iterations,
                batch_size=args.batch_size,
            )
        )
    report_path = write_benchmark_report(args.output_dir, rows)
    print(f"Saved benchmark report to {report_path}")


if __name__ == "__main__":
    main()
