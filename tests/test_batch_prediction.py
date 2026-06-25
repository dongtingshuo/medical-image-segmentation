import csv

import cv2
import numpy as np
import torch

from src.batch_prediction import batch_predict, collect_image_files
from src.model_unet import UNet


def test_collect_image_files_filters_supported_extensions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")

    files = collect_image_files(tmp_path, extensions=("jpg", "png"))

    assert [path.name for path in files] == ["a.jpg", "b.png"]


def test_batch_predict_writes_outputs_and_csv(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(2):
        image = np.zeros((72, 80, 3), dtype=np.uint8)
        image[..., 1] = 80 + index * 20
        cv2.imwrite(str(image_dir / f"sample_{index}.png"), image)

    config = {
        "data": {"image_size": 64},
        "model": {"model_name": "unet", "in_channels": 3, "out_channels": 1, "base_channels": 8},
    }
    model = UNet(in_channels=3, out_channels=1, base_channels=8)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": 1}, checkpoint_path)

    summary = batch_predict(
        input_dir=image_dir,
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "batch_outputs",
        threshold=0.5,
        device="cpu",
    )

    assert summary["total_images"] == 2
    assert summary["succeeded"] == 2
    csv_path = tmp_path / "batch_outputs" / "batch_predictions.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
    assert all((tmp_path / "batch_outputs" / f"sample_{index}_overlay.png").exists() for index in range(2))
