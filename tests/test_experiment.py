import csv

from src.experiment import EXPERIMENT_FIELDS, append_experiment_result


def test_experiment_csv_migrates_legacy_header(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    csv_path = output_dir / "experiment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_name", "best_val_loss"])
        writer.writeheader()
        writer.writerow({"experiment_name": "legacy", "best_val_loss": "0.2"})

    append_experiment_result(output_dir, {"experiment_name": "new", "best_val_loss": 0.1, "best_epoch": 2})

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == EXPERIMENT_FIELDS
    assert rows[0]["experiment_name"] == "legacy"
    assert rows[1]["best_epoch"] == "2"
