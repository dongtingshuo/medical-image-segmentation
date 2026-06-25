from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path


def mean_std_ci(values, confidence_z=1.96):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = confidence_z * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def bootstrap_mean_difference(a_values, b_values, iterations=1000, seed=42):
    a_values = [float(value) for value in a_values]
    b_values = [float(value) for value in b_values]
    if not a_values or not b_values:
        raise ValueError("Bootstrap inputs cannot be empty.")
    paired = len(a_values) == len(b_values)
    rng = random.Random(int(seed))
    differences = []
    if paired:
        n = len(a_values)
        for _ in range(int(iterations)):
            indices = [rng.randrange(n) for _ in range(n)]
            differences.append(statistics.fmean(b_values[index] - a_values[index] for index in indices))
    else:
        for _ in range(int(iterations)):
            sample_a = [a_values[rng.randrange(len(a_values))] for _ in range(len(a_values))]
            sample_b = [b_values[rng.randrange(len(b_values))] for _ in range(len(b_values))]
            differences.append(statistics.fmean(sample_b) - statistics.fmean(sample_a))
    differences.sort()
    low_index = int(0.025 * (len(differences) - 1))
    high_index = int(0.975 * (len(differences) - 1))
    observed = statistics.fmean(b_values) - statistics.fmean(a_values)
    return {
        "paired": paired,
        "iterations": int(iterations),
        "observed_difference": observed,
        "ci95_low": differences[low_index],
        "ci95_high": differences[high_index],
    }


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_metric_summary(rows, group_keys, metrics):
    grouped = {}
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for key, group_rows in sorted(grouped.items()):
        prefix = dict(zip(group_keys, key))
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in {None, ""}]
            if not values:
                continue
            summary_rows.append({**prefix, "metric": metric, **mean_std_ci(values)})
    return summary_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_statistical_report(rows, output_dir, title="Statistical Analysis"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(output_dir / "statistical_analysis.csv", rows)
    json_path = output_dir / "statistical_analysis.json"
    md_path = output_dir / "statistical_analysis.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        "| Group | Metric | N | Mean | Std | 95% CI Low | 95% CI High |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        group = " / ".join(str(value) for key, value in row.items() if key not in {"metric", "n", "mean", "std", "ci95_low", "ci95_high"})
        lines.append(
            f"| {group} | {row['metric']} | {row['n']} | {row['mean']:.6f} | {row['std']:.6f} | "
            f"{row['ci95_low']:.6f} | {row['ci95_high']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}
