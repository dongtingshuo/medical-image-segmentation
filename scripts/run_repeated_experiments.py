import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_suite import run_repeated_experiments  # noqa: E402
from src.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Train repeated random seeds and aggregate validation, test, and external metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output-root", default="outputs/repeated_experiments")
    parser.add_argument("--test-images-dir")
    parser.add_argument("--test-masks-dir")
    parser.add_argument("--external-images-dir")
    parser.add_argument("--external-masks-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_repeated_experiments(
        config=config,
        seeds=args.seeds,
        repo_root=ROOT,
        output_root=args.output_root,
        test_images_dir=args.test_images_dir,
        test_masks_dir=args.test_masks_dir,
        external_images_dir=args.external_images_dir,
        external_masks_dir=args.external_masks_dir,
    )
    print(f"Repeated experiment complete. Best seed: {result['best_seed']}")
    print(f"Summary: {Path(args.output_root).resolve() / 'summary.md'}")


if __name__ == "__main__":
    main()
