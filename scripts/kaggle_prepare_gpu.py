import argparse
import subprocess
import sys

P100_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
P100_TORCH_PACKAGES = [
    "torch==2.5.1",
    "torchvision==0.20.1",
]


def run(cmd):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def torch_report():
    import torch

    report = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_name": None,
        "capability": None,
        "cuda_runtime_ok": False,
        "error": None,
    }
    if torch.cuda.is_available():
        try:
            report["device_name"] = torch.cuda.get_device_name(0)
            report["capability"] = torch.cuda.get_device_capability(0)
            x = torch.randn(2, 3, 32, 32, device="cuda")
            conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            report["cuda_runtime_ok"] = bool(y.is_cuda and y.numel() > 0)
        except Exception as exc:  # noqa: BLE001
            report["error"] = repr(exc)
    return report


def print_report(report):
    print("Python:", report["python"], flush=True)
    print("PyTorch:", report["torch"], flush=True)
    print("CUDA available:", report["cuda_available"], flush=True)
    print("CUDA version:", report["cuda_version"], flush=True)
    print("GPU:", report["device_name"], flush=True)
    print("Capability:", report["capability"], flush=True)
    print("CUDA runtime smoke test:", report["cuda_runtime_ok"], flush=True)
    if report["error"]:
        print("CUDA runtime error:", report["error"], flush=True)


def is_pascal_or_older(report):
    capability = report.get("capability")
    return bool(capability and capability[0] < 7)


def install_p100_compatible_torch():
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "--index-url",
            P100_TORCH_INDEX,
            *P100_TORCH_PACKAGES,
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-if-needed", action="store_true")
    args = parser.parse_args()

    report = torch_report()
    print_report(report)
    if report["cuda_runtime_ok"]:
        print("Kaggle GPU is ready for PyTorch training.", flush=True)
        return

    if not report["cuda_available"]:
        raise SystemExit("CUDA is not available. Enable GPU in Kaggle Notebook Settings before formal training.")

    if is_pascal_or_older(report):
        message = (
            "Detected a Pascal-generation GPU such as Tesla P100. The current PyTorch build cannot run CUDA kernels "
            "on this device. Install a cu121 PyTorch build that still supports this GPU, then restart/rerun the notebook."
        )
        print(message, flush=True)
        if args.install_if_needed:
            install_p100_compatible_torch()
            print(
                "Installed compatible PyTorch packages. Restart the Kaggle session or rerun the notebook from the top.",
                flush=True,
            )
            return
        raise SystemExit(
            "GPU is present but unusable with current PyTorch. Run: "
            "python scripts/kaggle_prepare_gpu.py --install-if-needed"
        )

    raise SystemExit(
        "GPU is present but PyTorch CUDA smoke test failed. Check the CUDA runtime error above before formal training."
    )


if __name__ == "__main__":
    main()
