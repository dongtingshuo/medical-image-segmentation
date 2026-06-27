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


def stream_average(specs_by_name, names, images_dir, masks_dir, work_root, device, tta):
    running_path = Path(work_root) / "probability_mean.npy"
    temporary_path = Path(work_root) / "current_member.npy"
    running = None
    targets = None
    for index, name in enumerate(names, start=1):
        targets = cache_member(
            specs_by_name[name], images_dir, masks_dir, temporary_path, device, tta=tta
        )
        member = np.load(temporary_path, mmap_mode="r")
        if running is None:
            running = np.lib.format.open_memmap(running_path, mode="w+", dtype=np.float32, shape=member.shape)
            running[:] = member
        else:
            for start in range(0, len(member), 16):
                stop = min(start + 16, len(member))
                running[start:stop] += (member[start:stop] - running[start:stop]) / float(index)
        running.flush()
        del member
        temporary_path.unlink(missing_ok=True)
    return np.load(running_path, mmap_mode="r"), targets


def low_contrast_dice(probabilities, targets, dataset, manifest_path, threshold):
    if not manifest_path:
        return None
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        bins = {row["stem"]: row.get("contrast_bin") for row in csv.DictReader(handle)}
    stems = [image_path.stem for image_path, _ in dataset.pairs]
    indices = [index for index, stem in enumerate(stems) if bins.get(stem) == "0"]
    if not indices:
        return None
    return macro_metrics(probabilities[indices], targets[indices], threshold=threshold)["dice"]


def evaluate_variant(specs_by_name, variant, images_dir, masks_dir, work_root, device, manifest=None):
    names = variant.get("members") or [variant["member"]]
    probabilities, targets = stream_average(
        specs_by_name, names, images_dir, masks_dir, work_root, device, variant.get("tta", "none")
    )
    threshold = float(variant["threshold"])
    postprocess = variant.get("postprocess", {})
    if postprocess.get("enabled"):
        evaluated = postprocess_masks(
            probabilities,
            threshold,
            min_component_area=postprocess.get("min_component_area", 64),
            fill_holes=postprocess.get("fill_holes", True),
        )
        metrics = macro_metrics(evaluated * 0.999 + 0.0005, targets, threshold=0.5)
    else:
        evaluated = probabilities
        metrics = macro_metrics(probabilities, targets, threshold=threshold)
    dataset = SkinLesionDataset(images_dir, masks_dir, transform=None)
    low_contrast = low_contrast_dice(evaluated, targets, dataset, manifest, 0.5 if postprocess.get("enabled") else threshold)
    if low_contrast is not None:
        metrics["low_contrast_dice"] = low_contrast
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a locked v1.5 fast model and ensemble exactly once.")
    parser.add_argument("--members-json", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--test-images", required=True)
    parser.add_argument("--test-masks", required=True)
    parser.add_argument("--external-images", required=True)
    parser.add_argument("--external-masks", required=True)
    parser.add_argument("--test-manifest")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    completion_path = output_root / "evaluation_complete.json"
    if completion_path.exists() and not args.force:
        raise RuntimeError(f"Locked test/external evaluation already completed: {completion_path}")
    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    specs = load_specs(args.members_json)
    specs_by_name = {spec["name"]: spec for spec in specs}
    device = get_device(args.device)
    variants = {
        "fast": decision["fast"],
        "best_accuracy": {
            "members": decision["members"],
            "threshold": decision["threshold"],
            "tta": decision["tta"],
            "postprocess": decision["postprocess"],
        },
    }
    split_paths = {
        "test": (args.test_images, args.test_masks, args.test_manifest),
        "external": (args.external_images, args.external_masks, None),
    }
    results = {}
    for variant_name, variant in variants.items():
        results[variant_name] = {}
        for split, (images_dir, masks_dir, manifest) in split_paths.items():
            metrics = evaluate_variant(
                specs_by_name,
                variant,
                images_dir,
                masks_dir,
                output_root / "streaming" / variant_name / split,
                device,
                manifest=manifest,
            )
            criteria = ACCEPTANCE[split]
            metrics["accepted"] = all(metrics.get(key, -1.0) >= value for key, value in criteria.items())
            results[variant_name][split] = metrics
    results["publish_default"] = bool(results["fast"]["test"]["accepted"] and results["fast"]["external"]["accepted"])
    results["publish_best_accuracy"] = bool(
        results["best_accuracy"]["test"]["accepted"] and results["best_accuracy"]["external"]["accepted"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
