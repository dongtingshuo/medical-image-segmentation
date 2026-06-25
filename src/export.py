import hashlib
import json
from pathlib import Path

import torch

from src.inference import build_model_from_config
from src.utils import checkpoint_model_config, create_dirs, get_device, load_checkpoint, load_checkpoint_payload


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size_from_config(config):
    image_size = config.get("data", {}).get("image_size", config.get("image_size", 256))
    if isinstance(image_size, (list, tuple)):
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def load_model_for_export(config, checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    selected_device = get_device(device)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device=selected_device)
    model = build_model_from_config(config, checkpoint=checkpoint_payload).to(selected_device)
    expected_model_config = checkpoint_model_config(checkpoint_payload) or config.get("model", {})
    load_checkpoint(
        checkpoint_path,
        model,
        selected_device,
        expected_model_config=expected_model_config,
        checkpoint=checkpoint_payload,
    )
    model.eval()
    return model, selected_device, checkpoint_payload


def export_torchscript(model, dummy_input, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input, strict=False)
        traced = torch.jit.freeze(traced)
    traced.save(str(output_path))
    return output_path


def export_onnx(model, dummy_input, output_path, opset=17, dynamic_batch=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}, "logits": {0: "batch"}}
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=int(opset),
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "onnx":
            raise RuntimeError(
                "ONNX export requires the `onnx` package. Install requirements.txt or run `pip install onnx`."
            ) from exc
        raise
    return output_path


def export_model(
    config,
    checkpoint_path,
    output_dir="exports",
    formats=("torchscript", "onnx"),
    device="cpu",
    opset=17,
    batch_size=1,
    dynamic_batch=True,
):
    output_dir = Path(output_dir)
    create_dirs(output_dir)
    model, selected_device, checkpoint_payload = load_model_for_export(config, checkpoint_path, device=device)
    height, width = image_size_from_config(config)
    dummy_input = torch.randn(int(batch_size), 3, height, width, device=selected_device)

    files = {}
    normalized_formats = {str(fmt).lower() for fmt in formats}
    if "torchscript" in normalized_formats:
        files["torchscript"] = export_torchscript(model, dummy_input, output_dir / "model_torchscript.pt")
    if "onnx" in normalized_formats:
        files["onnx"] = export_onnx(
            model,
            dummy_input,
            output_dir / "model.onnx",
            opset=opset,
            dynamic_batch=dynamic_batch,
        )
    unsupported = normalized_formats - {"torchscript", "onnx"}
    if unsupported:
        raise ValueError(f"Unsupported export formats: {sorted(unsupported)}")

    manifest = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "device": str(selected_device),
        "input_shape": [int(batch_size), 3, height, width],
        "formats": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in files.items()
        },
        "model": checkpoint_model_config(checkpoint_payload) or config.get("model", {}),
        "outputs_are_logits": True,
        "postprocessing": "Apply sigmoid, then threshold during inference.",
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
