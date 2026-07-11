import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_ensemble_v1_5 import cache_member, load_specs  # noqa: E402
from src.dataset import SkinLesionDataset  # noqa: E402
from src.ensemble_v15 import macro_metrics, postprocess_masks  # noqa: E402
from src.utils import get_device  # noqa: E402

ACCEPTANCE = {
    "test": {"dice": 0.884766, "boundary_f1": 0.437615, "low_contrast_dice": 0.852973},
    "external": {"dice": 0.924386, "boundary_f1": 0.593236},
}


def _teacher_probability(specs, members, family_weights, cache_root, images_dir, masks_dir, device, tta="none"):
    selected = {spec["name"]: spec for spec in specs}
    family_values = {"unetpp": [], "segformer": []}
    targets = None
    for name in members:
        spec = selected[name]
        path = Path(cache_root) / f"{name}.npy"
        targets = cache_member(spec, images_dir, masks_dir, path, device, tta=tta)
        family_values[spec["architecture"]].append(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
    if any(len(values) != 5 for values in family_values.values()):
        raise ValueError("Locked v1.6 ensemble must contain ten teachers.")
    probability = sum(float(family_weights[family]) * np.mean(values, axis=0, dtype=np.float32) for family, values in family_values.items())
    return probability, targets


def _low_contrast_dice(probability, targets, images_dir, masks_dir, manifest_path, threshold):
    if not manifest_path:
        return None
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        bins = {row["stem"]: row.get("contrast_bin") for row in csv.DictReader(handle)}
    dataset = SkinLesionDataset(images_dir, masks_dir, transform=None)
    indices = [index for index, (image, _) in enumerate(dataset.pairs) if bins.get(image.stem) == "0"]
    return None if not indices else macro_metrics(probability[indices], targets[indices], threshold=threshold)["dice"]


def _evaluate(specs, decision, variant, images_dir, masks_dir, cache_root, device, manifest_path=None):
    if variant == "best_accuracy":
        probability, targets = _teacher_probability(specs, decision["members"], decision["family_weights"], cache_root, images_dir, masks_dir, device, tta=decision.get("tta", "none"))
        threshold = float(decision["threshold"])
    else:
        member = next(spec for spec in specs if spec["name"] == decision["fast"]["member"])
        path = Path(cache_root) / f"{member['name']}.npy"
        targets = cache_member(member, images_dir, masks_dir, path, device, tta="none")
        probability = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        threshold = float(decision["fast"]["threshold"])
    postprocess = decision.get("postprocess", {}) if variant == "best_accuracy" else decision["fast"].get("postprocess", {})
    evaluated = probability
    evaluated_threshold = threshold
    if postprocess.get("enabled"):
        evaluated = postprocess_masks(probability, threshold, min_component_area=64, fill_holes=True) * 0.999 + 0.0005
        evaluated_threshold = 0.5
    metrics = macro_metrics(evaluated, targets, threshold=evaluated_threshold)
    low_contrast = _low_contrast_dice(evaluated, targets, images_dir, masks_dir, manifest_path, evaluated_threshold)
    if low_contrast is not None:
        metrics["low_contrast_dice"] = low_contrast
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate an already locked v1.6 plan exactly once.")
    parser.add_argument("--members-json", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--test-images", required=True)
    parser.add_argument("--test-masks", required=True)
    parser.add_argument("--external-images", required=True)
    parser.add_argument("--external-masks", required=True)
    parser.add_argument("--test-manifest")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    completion = output_root / "evaluation_complete.json"
    if completion.exists():
        raise RuntimeError(f"Locked test/external evaluation already completed: {completion}")
    specs = load_specs(args.members_json)
    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    device = get_device(args.device)
    results = {"fast": {}, "best_accuracy": {}}
    for variant in results:
        for split, paths in {
            "test": (args.test_images, args.test_masks, args.test_manifest),
            "external": (args.external_images, args.external_masks, None),
        }.items():
            metrics = _evaluate(specs, decision, variant, paths[0], paths[1], output_root / "streaming" / variant / split, device, paths[2])
            metrics["accepted"] = all(metrics.get(key, -1.0) >= value for key, value in ACCEPTANCE[split].items())
            results[variant][split] = metrics
    results["publish_default"] = bool(results["fast"]["test"]["accepted"] and results["fast"]["external"]["accepted"])
    results["publish_best_accuracy"] = bool(results["best_accuracy"]["test"]["accepted"] and results["best_accuracy"]["external"]["accepted"])
    output_root.mkdir(parents=True, exist_ok=True)
    completion.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
