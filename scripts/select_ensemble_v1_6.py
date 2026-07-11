import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_ensemble_v1_5 import cache_member, load_specs  # noqa: E402
from src.ensemble_v15 import macro_metrics, postprocess_masks, search_macro_threshold, write_decision  # noqa: E402
from src.utils import get_device  # noqa: E402


def weighted_teacher_probabilities(specs, cache_root, images_dir, masks_dir, family_weights, device, tta="none"):
    by_family = {"unetpp": [], "segformer": []}
    targets = None
    for spec in specs:
        if spec.get("kind") != "teacher":
            continue
        family = spec["architecture"]
        path = Path(cache_root) / f"{spec['name']}.npy"
        targets = cache_member(spec, images_dir, masks_dir, path, device, tta=tta)
        by_family[family].append(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
    if any(len(by_family[family]) != 5 for family in by_family):
        raise ValueError("v1.6 best_accuracy requires all five teacher folds from both architecture families.")
    family_means = {family: np.mean(values, axis=0, dtype=np.float32) for family, values in by_family.items()}
    probability = sum(float(family_weights[family]) * family_means[family] for family in family_means)
    return probability, targets


def main():
    parser = argparse.ArgumentParser(description="Calibrate an OOF-locked v1.6 teacher ensemble on ISIC17 validation.")
    parser.add_argument("--members-json", required=True)
    parser.add_argument("--family-selection", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    specs = load_specs(args.members_json)
    family_selection = json.loads(Path(args.family_selection).read_text(encoding="utf-8"))
    family_weights = {name: float(value) for name, value in family_selection["family_weights"].items()}
    if set(family_weights) != {"unetpp", "segformer"} or not np.isclose(sum(family_weights.values()), 1.0):
        raise ValueError("OOF family weights must contain unetpp and segformer and sum to one.")
    probability, targets = weighted_teacher_probabilities(
        specs,
        output_root / "validation_probability_cache" / family_selection.get("tta", "none"),
        args.images_dir,
        args.masks_dir,
        family_weights,
        get_device(args.device),
        tta=family_selection.get("tta", "none"),
    )
    postprocess = family_selection.get("postprocess", {"enabled": False})
    if postprocess.get("enabled"):
        threshold_rows = []
        for threshold in np.arange(0.20, 0.7001, 0.01):
            cleaned = postprocess_masks(probability, threshold, min_component_area=64, fill_holes=True)
            threshold_rows.append({"threshold": float(round(threshold, 6)), **macro_metrics(cleaned * 0.999 + 0.0005, targets, threshold=0.5)})
        metrics = max(threshold_rows, key=lambda row: (row["dice"], row["boundary_f1"]))
    else:
        metrics, threshold_rows = search_macro_threshold(probability, targets, start=0.20, stop=0.70, step=0.01)
    fast_architecture = max(family_weights, key=lambda name: (family_weights[name], name))
    decision = {
        "members": [spec["name"] for spec in specs if spec.get("kind") == "teacher"],
        "family_weights": family_weights,
        "threshold": metrics["threshold"],
        "tta": family_selection.get("tta", "none"),
        "postprocess": postprocess,
        "validation_metrics": metrics,
        "oof_family_selection": family_selection["oof_metrics"],
        "fast": {"member": f"student-{fast_architecture}", "threshold": 0.5, "tta": "none", "postprocess": {"enabled": False, "min_component_area": 0, "fill_holes": False}},
    }
    write_decision(output_root / "locked_decision.json", decision)
    (output_root / "threshold_search.json").write_text(json.dumps(threshold_rows, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
