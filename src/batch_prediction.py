import csv
import json
import re
import time
from pathlib import Path

from src.inference import predict_file
from src.utils import create_dirs

DEFAULT_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")


def collect_image_files(input_dir, extensions=DEFAULT_IMAGE_EXTENSIONS, recursive=False):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    normalized_exts = {f".{str(ext).lower().lstrip('.')}" for ext in extensions}
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in normalized_exts)


def safe_prefix(image_path, input_dir):
    relative = Path(image_path).relative_to(input_dir)
    stem = relative.with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", stem)


def write_batch_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "status",
        "error",
        "lesion_ratio",
        "inference_time",
        "device",
        "model_name",
        "checkpoint_epoch",
        "output_image",
        "pred_mask",
        "overlay",
        "lesion_ratio_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def batch_predict(
    input_dir,
    config,
    checkpoint_path,
    output_dir,
    threshold=0.5,
    device="auto",
    recursive=False,
    extensions=DEFAULT_IMAGE_EXTENSIONS,
    continue_on_error=True,
    model_name_override=None,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    create_dirs(output_dir)
    image_files = collect_image_files(input_dir, extensions=extensions, recursive=recursive)
    if not image_files:
        raise FileNotFoundError(
            f"No images found in {input_dir}. Supported extensions: {', '.join(sorted(extensions))}"
        )

    rows = []
    started_at = time.time()
    for image_path in image_files:
        prefix = safe_prefix(image_path, input_dir)
        try:
            result = predict_file(
                image_path=image_path,
                config=config,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                threshold=threshold,
                device=device,
                model_name_override=model_name_override,
            )
            generated_paths = {}
            for key, path in result["paths"].items():
                if key == "lesion_ratio":
                    continue
                path = Path(path)
                renamed = output_dir / path.name.replace(image_path.stem, prefix, 1)
                if renamed != path:
                    path.replace(renamed)
                generated_paths[key] = renamed
            lesion_ratio_file = output_dir / f"{prefix}_lesion_ratio.txt"
            old_lesion_ratio_file = output_dir / f"{image_path.stem}_lesion_ratio.txt"
            if old_lesion_ratio_file.exists() and old_lesion_ratio_file != lesion_ratio_file:
                old_lesion_ratio_file.replace(lesion_ratio_file)
            rows.append(
                {
                    "image_path": str(image_path),
                    "status": "ok",
                    "error": "",
                    "lesion_ratio": f"{float(result['lesion_ratio']):.6f}",
                    "inference_time": f"{float(result['inference_time']):.6f}",
                    "device": result["device"],
                    "model_name": result["model_name"],
                    "checkpoint_epoch": result["checkpoint_epoch"] if result["checkpoint_epoch"] is not None else "",
                    "output_image": str(generated_paths.get("image", "")),
                    "pred_mask": str(generated_paths.get("pred_mask", "")),
                    "overlay": str(generated_paths.get("overlay", "")),
                    "lesion_ratio_file": str(lesion_ratio_file),
                }
            )
        except Exception as exc:  # noqa: BLE001
            if not continue_on_error:
                raise
            rows.append(
                {
                    "image_path": str(image_path),
                    "status": "error",
                    "error": str(exc),
                    "lesion_ratio": "",
                    "inference_time": "",
                    "device": "",
                    "model_name": "",
                    "checkpoint_epoch": "",
                    "output_image": "",
                    "pred_mask": "",
                    "overlay": "",
                    "lesion_ratio_file": "",
                }
            )

    csv_path = write_batch_csv(output_dir / "batch_predictions.csv", rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "threshold": float(threshold),
        "recursive": bool(recursive),
        "total_images": len(image_files),
        "succeeded": sum(row["status"] == "ok" for row in rows),
        "failed": sum(row["status"] != "ok" for row in rows),
        "elapsed_time": time.time() - started_at,
        "csv_path": str(csv_path),
    }
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
