import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SkinLesionDataset, get_val_transforms  # noqa: E402
from src.ensemble_v15 import (  # noqa: E402
    greedy_select_members,
    macro_metrics,
    postprocess_masks,
    search_macro_threshold,
    tta_probabilities,
    write_decision,
)
from src.inference import build_model_from_config  # noqa: E402
from src.utils import (  # noqa: E402
    checkpoint_model_config,
    get_device,
    load_checkpoint,
    load_checkpoint_payload,
    load_config,
)


def load_specs(path):
    specs = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "config", "checkpoint"}
    for spec in specs:
        if not required <= set(spec):
            raise ValueError(f"Member spec requires {sorted(required)}: {spec}")
    return specs


@torch.no_grad()
def cache_member(spec, images_dir, masks_dir, cache_path, device, tta="none"):
    config = load_config(spec["config"])
    dataset = SkinLesionDataset(images_dir, masks_dir, transform=get_val_transforms(config))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    checkpoint = load_checkpoint_payload(spec["checkpoint"], device=device)
    model = build_model_from_config(config, checkpoint=checkpoint).to(device)
    expected = checkpoint_model_config(checkpoint) or config.get("model", {})
    load_checkpoint(spec["checkpoint"], model, device, expected_model_config=expected, checkpoint=checkpoint)
    model.eval()
    probabilities, targets = [], []
    for images, masks in loader:
        probability = tta_probabilities(model, images.to(device, non_blocking=True), mode=tta)
        probabilities.append(probability.cpu().numpy().astype(np.float16))
        targets.append(masks.numpy().astype(np.uint8))
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.concatenate(probabilities))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(targets)


def average_selected(specs, selected, images_dir, masks_dir, cache_root, device, tta):
    running = None
    targets = None
    selected_specs = {spec["name"]: spec for spec in specs}
    for index, name in enumerate(selected, start=1):
        cache_path = Path(cache_root) / tta / f"{name}.npy"
        targets = cache_member(selected_specs[name], images_dir, masks_dir, cache_path, device, tta=tta)
        probability = np.asarray(np.load(cache_path, mmap_mode="r"), dtype=np.float32)
        running = probability.copy() if running is None else running + (probability - running) / float(index)
    return running, targets


def main():
    parser = argparse.ArgumentParser(description="Lock the v1.5 validation ensemble without touching test data.")
    parser.add_argument("--members-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-members", type=int, default=5)
    parser.add_argument("--min-improvement", type=float, default=0.0005)
    args = parser.parse_args()

    specs = load_specs(args.members_json)
    output_root = Path(args.output_root)
    cache_root = output_root / "validation_probability_cache"
    device = get_device(args.device)
    targets = None
    member_paths = {}
    individual_metrics = {}
    for spec in specs:
        path = cache_root / "none" / f"{spec['name']}.npy"
        targets = cache_member(spec, args.images_dir, args.masks_dir, path, device, tta="none")
        member_paths[spec["name"]] = path
        individual_metrics[spec["name"]], _ = search_macro_threshold(
            np.load(path, mmap_mode="r"), targets
        )
    np.save(output_root / "validation_targets.npy", targets)
    selected, probabilities, history = greedy_select_members(
        member_paths, targets, min_improvement=args.min_improvement, max_members=args.max_members
    )
    base_metrics, _ = search_macro_threshold(probabilities, targets)
    chosen_tta = "none"
    chosen_probability = probabilities
    chosen_metrics = base_metrics
    for tta in ["flip", "multiscale_flip"]:
        candidate_probability, candidate_targets = average_selected(
            specs, selected, args.images_dir, args.masks_dir, cache_root, device, tta
        )
        metrics, _ = search_macro_threshold(candidate_probability, candidate_targets)
        if (
            metrics["composite"] >= chosen_metrics["composite"] + 0.001
            and metrics["boundary_f1"] >= chosen_metrics["boundary_f1"]
        ):
            chosen_tta, chosen_probability, chosen_metrics = tta, candidate_probability, metrics

    cleaned = postprocess_masks(chosen_probability, chosen_metrics["threshold"])
    cleaned_metrics = macro_metrics(cleaned * 0.999 + 0.0005, targets, threshold=0.5)
    postprocess = {"enabled": False, "min_component_area": 0, "fill_holes": False}
    if (
        cleaned_metrics["composite"] >= chosen_metrics["composite"] + args.min_improvement
        and cleaned_metrics["boundary_f1"] >= chosen_metrics["boundary_f1"]
    ):
        postprocess = {"enabled": True, "min_component_area": 64, "fill_holes": True}
        chosen_metrics = {"threshold": chosen_metrics["threshold"], **cleaned_metrics}

    student_names = [spec["name"] for spec in specs if spec.get("kind") == "student"]
    fast_candidates = student_names or list(member_paths)
    fast_member = max(
        fast_candidates,
        key=lambda name: (individual_metrics[name]["composite"], individual_metrics[name]["dice"], name),
    )

    decision = {
        "fast": {
            "member": fast_member,
            "threshold": individual_metrics[fast_member]["threshold"],
            "tta": "none",
            "postprocess": {"enabled": False, "min_component_area": 0, "fill_holes": False},
            "validation_metrics": individual_metrics[fast_member],
        },
        "members": selected,
        "threshold": chosen_metrics["threshold"],
        "tta": chosen_tta,
        "postprocess": postprocess,
        "validation_metrics": chosen_metrics,
        "selection_history": history,
    }
    write_decision(output_root / "locked_decision.json", decision)
    with (output_root / "selection_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
