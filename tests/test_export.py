import torch

from src.export import export_model
from src.model_unet import UNet


def test_export_torchscript_runs_inference(tmp_path):
    config = {
        "data": {"image_size": 64},
        "model": {"model_name": "unet", "in_channels": 3, "out_channels": 1, "base_channels": 8},
    }
    model = UNet(in_channels=3, out_channels=1, base_channels=8)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": 1}, checkpoint_path)

    manifest = export_model(
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "exports",
        formats=("torchscript",),
        device="cpu",
    )

    exported_path = manifest["formats"]["torchscript"]["path"]
    exported = torch.jit.load(exported_path, map_location="cpu")
    with torch.no_grad():
        logits = exported(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 1, 64, 64)
    assert manifest["outputs_are_logits"] is True
