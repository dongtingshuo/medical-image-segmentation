import csv
from pathlib import Path


EXPERIMENT_FIELDS = [
    "experiment_name",
    "model_name",
    "device",
    "image_size",
    "batch_size",
    "epochs",
    "requested_epochs",
    "best_epoch",
    "lr",
    "loss_name",
    "augmentation_enabled",
    "best_val_loss",
    "val_loss_at_best_epoch",
    "best_dice",
    "best_iou",
    "precision",
    "recall",
    "specificity",
    "boundary_f1",
    "checkpoint_path",
    "checkpoint_format_version",
    "training_time",
]


def append_experiment_result(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment_results.csv"
    exists = csv_path.exists()
    if exists:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if old_fields != EXPERIMENT_FIELDS:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
                writer.writeheader()
                writer.writerows({field: row.get(field, "") for field in EXPERIMENT_FIELDS} for row in old_rows)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in EXPERIMENT_FIELDS})
    return csv_path
