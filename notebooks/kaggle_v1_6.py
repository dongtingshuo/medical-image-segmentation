import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"


def run(command, cwd=None):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def resolve_runtime_roots(input_root=None, working_root=None):
    cwd = Path.cwd().resolve()
    working = Path(working_root).resolve() if working_root else next(
        (path for path in (cwd, *cwd.parents) if path.name == "working" and path.parent.name == "kaggle"), cwd
    )
    inputs = Path(input_root).resolve() if input_root else next(
        (path / "input" for path in (cwd, *cwd.parents) if (path / "input").is_dir()), working.parent / "input"
    )
    return inputs, working


def resolve_dataset(input_root, reference, label):
    slug = str(reference).split("/")[-1]
    direct = Path(input_root) / slug
    if direct.exists():
        return direct
    matches = [path for path in Path(input_root).rglob(slug) if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot resolve {label} dataset `{reference}`: {matches}")
    return matches[0]


def resolve_pair_roots(root, images_relative=None, masks_relative=None, label="dataset"):
    if bool(images_relative) != bool(masks_relative):
        raise ValueError(f"Provide both {label} image and mask relative paths, or neither.")
    if not images_relative:
        return root, root
    images, masks = Path(root) / images_relative, Path(root) / masks_relative
    if not images.exists() or not masks.exists():
        raise FileNotFoundError(f"Missing {label} paths: {images}, {masks}")
    return images, masks


def resolve_file(root, relative, filename, label):
    if relative:
        candidate = Path(root) / relative
        if candidate.exists():
            return candidate
    matches = [path for path in Path(root).rglob(filename) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot resolve {label} `{filename}` below {root}: {matches[:5]}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Kaggle entrypoint for v1.6 target-domain training.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--use-existing-repo", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--input-root", default=os.environ.get("KAGGLE_INPUT_PATH"))
    parser.add_argument("--working-root", default=os.environ.get("KAGGLE_WORKING_PATH"))
    parser.add_argument("--state-input")
    parser.add_argument("--allow-state-mismatch", action="store_true")
    parser.add_argument("--isic17-dataset", default="moon1570/isic-2017-train-val-test-images-and-masks")
    parser.add_argument("--isic18-dataset", default="tntiphan/isic-2018-task-1")
    parser.add_argument("--isic16-dataset", default="mahmudulhasantasin/isic-2016-original-dataset")
    parser.add_argument("--ph2-dataset", default="spacesurfer/ph2-dataset")
    parser.add_argument("--ham-images-dataset", default="kmader/skin-cancer-mnist-ham10000")
    parser.add_argument("--ham-masks-dataset", default="tschandl/ham10000-lesion-segmentations")
    parser.add_argument("--isic16-images-rel")
    parser.add_argument("--isic16-masks-rel")
    parser.add_argument("--ph2-images-rel")
    parser.add_argument("--ph2-masks-rel")
    parser.add_argument("--ham-images-rel")
    parser.add_argument("--ham-masks-rel")
    parser.add_argument("--ham-metadata-rel")
    args = parser.parse_args()

    input_root, working_root = resolve_runtime_roots(args.input_root, args.working_root)
    repository = working_root / "medical-image-segmentation"
    if not (args.use_existing_repo and repository.exists()):
        if repository.exists():
            shutil.rmtree(repository)
        run(["git", "clone", "--depth", "1", REPOSITORY_URL, repository])
    run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-r", "requirements-kaggle.txt"],
        cwd=repository,
    )
    # W&B is intentionally disabled for v1.6: no Secret lookup, import, sync, or run initialization.
    run([sys.executable, "scripts/kaggle_prepare_gpu.py", "--install-if-needed"], cwd=repository)
    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"], cwd=repository)
    isic17 = resolve_dataset(input_root, args.isic17_dataset, "ISIC17")
    isic18 = resolve_dataset(input_root, args.isic18_dataset, "ISIC18")
    isic16 = resolve_dataset(input_root, args.isic16_dataset, "ISIC16")
    ph2 = resolve_dataset(input_root, args.ph2_dataset, "PH2")
    ham_images = resolve_dataset(input_root, args.ham_images_dataset, "HAM10000 images")
    ham_masks = resolve_dataset(input_root, args.ham_masks_dataset, "HAM10000 masks")
    isic16_images, isic16_masks = resolve_pair_roots(isic16, args.isic16_images_rel, args.isic16_masks_rel, "ISIC16")
    ph2_images, ph2_masks = resolve_pair_roots(ph2, args.ph2_images_rel, args.ph2_masks_rel, "PH2")
    ham_image_root = ham_images / args.ham_images_rel if args.ham_images_rel else ham_images
    ham_mask_root = ham_masks / args.ham_masks_rel if args.ham_masks_rel else ham_masks
    ham_metadata = resolve_file(ham_images, args.ham_metadata_rel, "HAM10000_metadata.csv", "HAM10000 metadata")
    state_input = args.state_input or next((str(path) for path in sorted(input_root.rglob("v1_6_state.zip"))), None)
    if args.debug:
        run(
            [
                sys.executable,
                "scripts/debug_v1_6.py",
                "--output",
                working_root / "v1_6_debug_report.json",
                "--isic16-images",
                isic16_images,
                "--isic16-masks",
                isic16_masks,
                "--ph2-images",
                ph2_images,
                "--ph2-masks",
                ph2_masks,
                "--ham-images",
                ham_image_root,
                "--ham-masks",
                ham_mask_root,
                "--ham-metadata",
                ham_metadata,
            ],
            cwd=repository,
        )
        return
    command = [
        sys.executable, "scripts/run_v1_6_pipeline.py", "--config", "configs/kaggle_v1_6_debug.yaml" if args.debug else "configs/kaggle_v1_6.yaml",
        "--output-root", working_root / "research_v1_6", "--isic17-root", isic17, "--isic18-root", isic18,
        "--isic16-images", isic16_images, "--isic16-masks", isic16_masks, "--ph2-images", ph2_images, "--ph2-masks", ph2_masks,
        "--ham-images", ham_image_root, "--ham-masks", ham_mask_root, "--ham-metadata", ham_metadata,
    ]
    if state_input:
        command.extend(["--state-input", state_input])
    if args.allow_state_mismatch:
        command.append("--allow-state-mismatch")
    if args.prepare_only:
        command.append("--prepare-only")
    run(command, cwd=repository)


if __name__ == "__main__":
    main()
