import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation import (  # noqa: E402
    collect_sample_ids,
    prepare_external_split,
    prepare_internal_splits,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare paired ISIC train/val/test and external evaluation data.")
    parser.add_argument("--internal-root", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output-root", default="/kaggle/working/prepared_data")
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    internal_report = prepare_internal_splits(
        args.internal_root,
        output_root / "internal",
        image_size=args.image_size,
    )
    external_report = prepare_external_split(
        args.external_root,
        output_root / "external",
        excluded_ids=collect_sample_ids(args.internal_root),
        image_size=args.image_size,
    )
    print("Internal prepared splits:", internal_report["prepared_splits"])
    print(
        "External prepared pairs:",
        external_report["prepared_pairs"],
        "source split:",
        external_report["selected_source_split"],
        "excluded overlaps:",
        external_report["excluded_overlap_count"],
    )


if __name__ == "__main__":
    main()
