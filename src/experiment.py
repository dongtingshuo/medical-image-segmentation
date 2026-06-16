import csv
from pathlib import Path


EXPERIMENT_FIELDS = [
    "experiment_name",
    "model_name",
    "device",
    "image_size",
    "batch_size",
    "epochs",
    "lr",
    "loss_name",
    "augmentation_enabled",
    "best_val_loss",
    "best_dice",
    "best_iou",
    "precision",
    "recall",
    "checkpoint_path",
    "training_time",
]


def append_experiment_result(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment_results.csv"
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in EXPERIMENT_FIELDS})
    return csv_path

