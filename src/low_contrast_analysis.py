import csv
import json
from pathlib import Path

METRIC_COLUMNS = ("dice", "iou", "precision", "recall")


def read_single_csv_row(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def read_low_contrast_row(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Subgroup summary does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("group_field") == "contrast_tertile" and row.get("group") == "low":
            return row
    raise ValueError(f"No low contrast tertile row found in {path}")


def _float_or_empty(row, key):
    value = row.get(key, "")
    return "" if value in {"", None} else float(value)


def _format_metric(value):
    return "" if value == "" else f"{float(value):.6f}"


def collect_variant_results(variants_root, variants, splits=("test", "external"), baseline_variant="control_bce_dice"):
    variants_root = Path(variants_root)
    rows = []
    baseline_by_split = {}
    for variant in variants:
        variant_root = variants_root / variant
        for split in splits:
            overall = read_single_csv_row(variant_root / f"evaluation_{split}" / "metrics.csv")
            low_contrast = read_low_contrast_row(variant_root / f"subgroup_{split}" / "subgroup_summary.csv")
            row = {
                "variant": variant,
                "split": split,
                "threshold": _float_or_empty(overall, "threshold"),
                "overall_dice": _float_or_empty(overall, "dice"),
                "overall_iou": _float_or_empty(overall, "iou"),
                "overall_precision": _float_or_empty(overall, "precision"),
                "overall_recall": _float_or_empty(overall, "recall"),
                "low_contrast_samples": int(float(low_contrast["samples"])),
                "low_contrast_dice": _float_or_empty(low_contrast, "dice_mean"),
                "low_contrast_iou": _float_or_empty(low_contrast, "iou_mean"),
                "low_contrast_precision": _float_or_empty(low_contrast, "precision_mean"),
                "low_contrast_recall": _float_or_empty(low_contrast, "recall_mean"),
            }
            rows.append(row)
            if variant == baseline_variant:
                baseline_by_split[split] = row

    for row in rows:
        baseline = baseline_by_split.get(row["split"])
        if baseline is None:
            row["low_contrast_dice_delta"] = ""
            row["low_contrast_recall_delta"] = ""
            row["overall_dice_delta"] = ""
            continue
        row["low_contrast_dice_delta"] = row["low_contrast_dice"] - baseline["low_contrast_dice"]
        row["low_contrast_recall_delta"] = row["low_contrast_recall"] - baseline["low_contrast_recall"]
        row["overall_dice_delta"] = row["overall_dice"] - baseline["overall_dice"]
    return rows


def select_best_variant(rows, target_split="test", max_overall_dice_drop=0.01):
    candidates = [
        row
        for row in rows
        if row["split"] == target_split
        and row["low_contrast_dice_delta"] != ""
        and row["overall_dice_delta"] >= -float(max_overall_dice_drop)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["low_contrast_dice"], row["low_contrast_recall"], row["overall_dice"]))


def replacement_recommendation(best_row, dice_delta_threshold=0.02, recall_delta_threshold=0.03):
    if best_row is None:
        return {
            "recommend_replacement": False,
            "reason": "No candidate satisfied the overall Dice drop constraint.",
        }
    dice_delta = best_row["low_contrast_dice_delta"]
    recall_delta = best_row["low_contrast_recall_delta"]
    recommend = dice_delta >= dice_delta_threshold or recall_delta >= recall_delta_threshold
    return {
        "recommend_replacement": bool(recommend),
        "best_variant": best_row["variant"],
        "split": best_row["split"],
        "low_contrast_dice_delta": dice_delta,
        "low_contrast_recall_delta": recall_delta,
        "overall_dice_delta": best_row["overall_dice_delta"],
        "reason": (
            "Low-contrast improvement satisfies the configured threshold."
            if recommend
            else "Low-contrast improvement does not satisfy the configured threshold."
        ),
    }


def write_low_contrast_outputs(rows, output_dir, recommendation):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "low_contrast_comparison.csv"
    json_path = output_dir / "low_contrast_comparison.json"
    md_path = output_dir / "low_contrast_comparison.md"
    fieldnames = [
        "variant",
        "split",
        "threshold",
        "overall_dice",
        "overall_iou",
        "overall_precision",
        "overall_recall",
        "low_contrast_samples",
        "low_contrast_dice",
        "low_contrast_iou",
        "low_contrast_precision",
        "low_contrast_recall",
        "low_contrast_dice_delta",
        "low_contrast_recall_delta",
        "overall_dice_delta",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({"rows": rows, "recommendation": recommendation}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Low-Contrast v1.3 Comparison",
        "",
        f"- Recommend replacing default model: `{recommendation['recommend_replacement']}`",
        f"- Reason: {recommendation['reason']}",
    ]
    if recommendation.get("best_variant"):
        lines.append(f"- Best variant: `{recommendation['best_variant']}`")
    lines.extend(
        [
            "",
            "| Variant | Split | Overall Dice | Low-Contrast Dice | Low-Contrast Recall | Low-Contrast Dice Delta | Low-Contrast Recall Delta | Overall Dice Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["split"], -float(item["low_contrast_dice"]))):
        lines.append(
            f"| {row['variant']} | {row['split']} | {row['overall_dice']:.6f} | "
            f"{row['low_contrast_dice']:.6f} | {row['low_contrast_recall']:.6f} | "
            f"{_format_metric(row['low_contrast_dice_delta'])} | "
            f"{_format_metric(row['low_contrast_recall_delta'])} | "
            f"{_format_metric(row['overall_dice_delta'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}
