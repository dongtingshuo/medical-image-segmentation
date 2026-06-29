import argparse
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

REPOSITORY_URL = "https://github.com/dongtingshuo/medical-image-segmentation.git"


def run(command, cwd=None):
    command = [str(part) for part in command]
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def resolve_dataset(input_root, reference, label):
    if not reference:
        raise ValueError(f"{label} dataset reference is required.")
    slug = reference.split("/")[-1]
    direct = input_root / slug
    if direct.exists():
        return direct
    matches = [path for path in input_root.rglob(slug) if path.is_dir()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve {label} dataset `{reference}` below {input_root}; matches={matches}")


def resolve_pair_roots(root, images_relative, masks_relative, label):
    if bool(images_relative) != bool(masks_relative):
        raise ValueError(f"Provide both {label} image and mask relative paths, or neither.")
    if not images_relative:
        print(f"Recursively discovering {label} image/mask pairs below {root}", flush=True)
        return root, root
    images = root / images_relative
    masks = root / masks_relative
    if not images.exists() or not masks.exists():
        raise FileNotFoundError(f"{label} directories do not exist: images={images}, masks={masks}")
    return images, masks


def load_wandb_secrets():
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        api_key = client.get_secret("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        try:
            entity = client.get_secret("WANDB_ENTITY")
        except Exception:  # noqa: BLE001
            entity = None
        if entity:
            os.environ["WANDB_ENTITY"] = entity
        print("W&B secret loaded without writing it to disk or logs.", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WANDB_API_KEY is unavailable; runs will be stored offline: {exc}", flush=True)


def resolve_runtime_roots(input_root=None, working_root=None):
    cwd = Path.cwd().resolve()
    if working_root:
        resolved_working = Path(working_root).resolve()
    else:
        resolved_working = next(
            (path for path in (cwd, *cwd.parents) if path.name == "working" and path.parent.name == "kaggle"),
            cwd,
        )

    if input_root:
        resolved_input = Path(input_root).resolve()
    else:
        resolved_input = next(
            (path / "input" for path in (cwd, *cwd.parents) if (path / "input").is_dir()),
            resolved_working.parent / "input",
        )
    return resolved_input, resolved_working


def main():
    parser = argparse.ArgumentParser(description="Kaggle entrypoint for resumable v1.5 training.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-existing-repo", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--input-root", default=os.environ.get("KAGGLE_INPUT_PATH"))
    parser.add_argument("--working-root", default=os.environ.get("KAGGLE_WORKING_PATH"))
    parser.add_argument("--isic17-dataset", default=os.environ.get("V15_ISIC17_DATASET", "moon1570/isic-2017-train-val-test-images-and-masks"))
    parser.add_argument("--isic18-dataset", default=os.environ.get("V15_ISIC18_DATASET", "tntiphan/isic-2018-task-1"))
    parser.add_argument(
        "--isic16-dataset",
        default=os.environ.get("V15_ISIC16_DATASET", "mahmudulhasantasin/isic-2016-original-dataset"),
    )
    parser.add_argument("--ph2-dataset", default=os.environ.get("V15_PH2_DATASET", "spacesurfer/ph2-dataset"))
    parser.add_argument("--isic16-images-rel", default=os.environ.get("V15_ISIC16_IMAGES_REL"))
    parser.add_argument("--isic16-masks-rel", default=os.environ.get("V15_ISIC16_MASKS_REL"))
    parser.add_argument("--ph2-images-rel", default=os.environ.get("V15_PH2_IMAGES_REL"))
    parser.add_argument("--ph2-masks-rel", default=os.environ.get("V15_PH2_MASKS_REL"))
    parser.add_argument("--state-input")
    parser.add_argument("--allow-state-mismatch", action="store_true")
    args = parser.parse_args()

    input_root, working_root = resolve_runtime_roots(args.input_root, args.working_root)
    repository_root = working_root / "medical-image-segmentation"
    if args.use_existing_repo and repository_root.exists():
        print(f"Using existing repository: {repository_root}", flush=True)
    else:
        if repository_root.exists():
            shutil.rmtree(repository_root)
        run(["git", "clone", "--depth", "1", REPOSITORY_URL, repository_root])

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], cwd=repository_root)
    wandb_version = version("wandb")
    if wandb_version != "0.22.3":
        raise RuntimeError(f"Expected wandb 0.22.3 for new API keys, found {wandb_version}")
    print(f"W&B SDK: {wandb_version}", flush=True)
    load_wandb_secrets()
    run([sys.executable, "scripts/kaggle_prepare_gpu.py", "--install-if-needed"], cwd=repository_root)
    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"], cwd=repository_root)

    isic17_root = resolve_dataset(input_root, args.isic17_dataset, "ISIC 2017")
    isic18_root = resolve_dataset(input_root, args.isic18_dataset, "ISIC 2018")
    isic16_root = resolve_dataset(input_root, args.isic16_dataset, "ISIC 2016")
    ph2_root = resolve_dataset(input_root, args.ph2_dataset, "PH2")
    isic16_images, isic16_masks = resolve_pair_roots(
        isic16_root, args.isic16_images_rel, args.isic16_masks_rel, "ISIC 2016"
    )
    ph2_images, ph2_masks = resolve_pair_roots(ph2_root, args.ph2_images_rel, args.ph2_masks_rel, "PH2")
    state_input = args.state_input
    if not state_input:
        state_candidates = sorted(input_root.rglob("v1_5_state.zip"))
        state_input = str(state_candidates[-1]) if state_candidates else None

    command = [
        sys.executable,
        "scripts/run_v1_5_pipeline.py",
        "--config",
        "configs/kaggle_v1_5_debug.yaml" if args.debug else "configs/kaggle_v1_5.yaml",
        "--output-root",
        working_root / "research_v1_5",
        "--isic17-root",
        isic17_root,
        "--isic18-root",
        isic18_root,
        "--isic16-images",
        isic16_images,
        "--isic16-masks",
        isic16_masks,
        "--ph2-images",
        ph2_images,
        "--ph2-masks",
        ph2_masks,
    ]
    if state_input:
        command.extend(["--state-input", state_input])
    if args.allow_state_mismatch:
        command.append("--allow-state-mismatch")
    if args.debug:
        command.extend(["--runtime-minutes", "25", "--reserve-minutes", "3"])
    run(command, cwd=repository_root)


if __name__ == "__main__":
    main()
