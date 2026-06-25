import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cross_validation import (  # noqa: E402
    create_kfold_splits,
    materialize_fold_directories,
    paired_stems,
    write_folds,
)


def main():
    parser = argparse.ArgumentParser(description="Create reproducible k-fold image/mask directories.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()

    stems = paired_stems(args.images_dir, args.masks_dir)
    folds = create_kfold_splits(stems, k=args.k, seed=args.seed)
    output_root = Path(args.output_root)
    folds_path = write_folds(
        output_root / "folds.json",
        folds,
        metadata={"k": args.k, "seed": args.seed, "samples": len(stems)},
    )
    if args.materialize:
        materialize_fold_directories(args.images_dir, args.masks_dir, folds, output_root)
    print(f"Saved folds to {folds_path}")


if __name__ == "__main__":
    main()
