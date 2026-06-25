import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.low_contrast_analysis import (  # noqa: E402
    collect_variant_results,
    replacement_recommendation,
    select_best_variant,
    write_low_contrast_outputs,
)


def main():
    parser = argparse.ArgumentParser(description="Compare low-contrast v1.3 experiment variants.")
    parser.add_argument("--variants-root", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "external"])
    parser.add_argument("--baseline-variant", default="control_bce_dice")
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--max-overall-dice-drop", type=float, default=0.01)
    parser.add_argument("--dice-delta-threshold", type=float, default=0.02)
    parser.add_argument("--recall-delta-threshold", type=float, default=0.03)
    parser.add_argument("--output-dir", default="outputs/low_contrast_comparison")
    args = parser.parse_args()

    rows = collect_variant_results(
        variants_root=args.variants_root,
        variants=args.variants,
        splits=args.splits,
        baseline_variant=args.baseline_variant,
    )
    best = select_best_variant(
        rows,
        target_split=args.target_split,
        max_overall_dice_drop=args.max_overall_dice_drop,
    )
    recommendation = replacement_recommendation(
        best,
        dice_delta_threshold=args.dice_delta_threshold,
        recall_delta_threshold=args.recall_delta_threshold,
    )
    outputs = write_low_contrast_outputs(rows, args.output_dir, recommendation)
    print(f"Saved low-contrast comparison report to {outputs['markdown']}")
    print(f"Recommend replacement: {recommendation['recommend_replacement']}")
    if recommendation.get("best_variant"):
        print(f"Best variant: {recommendation['best_variant']}")


if __name__ == "__main__":
    main()
