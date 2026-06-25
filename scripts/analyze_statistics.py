import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statistical_analysis import group_metric_summary, read_csv_rows, write_statistical_report  # noqa: E402


def write_combined_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    return path


def infer_group_keys(rows, input_count):
    if not rows:
        return ["source"]
    preferred = []
    if input_count > 1:
        preferred.append("source")
    for key in ("encoder", "split", "group_field", "group"):
        if any(row.get(key) not in {None, ""} for row in rows):
            preferred.append(key)
    return preferred or ["source"]


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment metrics with mean/std/95% CI.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/statistical_analysis")
    parser.add_argument("--group-by", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="+", default=["dice", "iou", "precision", "recall", "loss"])
    args = parser.parse_args()

    combined = []
    for input_path in args.inputs:
        path = Path(input_path)
        for row in read_csv_rows(path):
            item = dict(row)
            item.setdefault("source", path.stem)
            combined.append(item)
    group_keys = args.group_by if args.group_by else infer_group_keys(combined, len(args.inputs))
    metrics = [metric for metric in args.metrics if any(row.get(metric) not in {None, ""} for row in combined)]
    if not metrics:
        raise ValueError(f"None of the requested metrics are available: {args.metrics}")
    rows = group_metric_summary(combined, group_keys, metrics)
    output_dir = Path(args.output_dir)
    write_combined_csv(output_dir / "combined_metrics.csv", combined)
    outputs = write_statistical_report(rows, output_dir, title="Research Statistical Summary")
    print(f"Saved statistical report to {outputs['markdown']}")


if __name__ == "__main__":
    main()
