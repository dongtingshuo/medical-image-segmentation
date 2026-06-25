import csv

from src.low_contrast_analysis import (
    collect_variant_results,
    replacement_recommendation,
    select_best_variant,
    write_low_contrast_outputs,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_variant(root, variant, split, overall_dice, low_dice, low_recall):
    variant_root = root / variant
    write_csv(
        variant_root / f"evaluation_{split}" / "metrics.csv",
        [
            {
                "split": split,
                "samples": 10,
                "threshold": 0.4,
                "loss": 0.1,
                "dice": overall_dice,
                "iou": 0.8,
                "precision": 0.9,
                "recall": 0.85,
            }
        ],
    )
    write_csv(
        variant_root / f"subgroup_{split}" / "subgroup_summary.csv",
        [
            {
                "group_field": "contrast_tertile",
                "group": "low",
                "samples": 4,
                "dice_mean": low_dice,
                "dice_std": 0.0,
                "iou_mean": 0.7,
                "iou_std": 0.0,
                "precision_mean": 0.88,
                "precision_std": 0.0,
                "recall_mean": low_recall,
                "recall_std": 0.0,
            }
        ],
    )


def test_low_contrast_comparison_selects_best_variant(tmp_path):
    variants_root = tmp_path / "variants"
    add_variant(variants_root, "control_bce_dice", "test", 0.86, 0.82, 0.79)
    add_variant(variants_root, "contrast_aug_bce_dice", "test", 0.855, 0.845, 0.83)
    add_variant(variants_root, "contrast_aug_tversky", "test", 0.82, 0.86, 0.86)

    rows = collect_variant_results(
        variants_root,
        ["control_bce_dice", "contrast_aug_bce_dice", "contrast_aug_tversky"],
        splits=["test"],
    )
    best = select_best_variant(rows, target_split="test", max_overall_dice_drop=0.01)
    recommendation = replacement_recommendation(best)
    outputs = write_low_contrast_outputs(rows, tmp_path / "comparison", recommendation)

    assert best["variant"] == "contrast_aug_bce_dice"
    assert recommendation["recommend_replacement"] is True
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
