import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks/kaggle_aggressive_v1_4.py"
SPEC = importlib.util.spec_from_file_location("kaggle_aggressive_v1_4", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tta_failures_are_recorded_without_stopping_other_variants(tmp_path, monkeypatch):
    commands = []

    def fake_run_logged(command, log_path, cwd=None):
        commands.append([str(part) for part in command])
        if "tta_best_accuracy_test" in str(log_path):
            raise RuntimeError("simulated evaluation failure")

    monkeypatch.setattr(MODULE, "run_logged", fake_run_logged)
    failures = MODULE.evaluate_tta_variants(
        tmp_path / "config.yaml",
        tmp_path / "best_model.pth",
        tmp_path / "experiment",
        threshold=0.25,
    )

    assert len(commands) == 6
    assert len(failures) == 1
    assert failures[0]["variant"] == "best_accuracy"
    best_accuracy_commands = [command for command in commands if "0.875,1.0,1.125" in command]
    assert all(command[command.index("--batch-size") + 1] == "1" for command in best_accuracy_commands)
    assert (tmp_path / "experiment/tta_failures.json").exists()


def test_analysis_output_requires_macro_schema(tmp_path):
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("dice\n0.87\n", encoding="utf-8")
    current = tmp_path / "current.csv"
    current.write_text(
        "aggregation,dice,micro_dice\nmacro_per_image,0.86,0.87\n",
        encoding="utf-8",
    )

    assert not MODULE.analysis_output_is_current(legacy)
    assert MODULE.analysis_output_is_current(current)


def test_summary_includes_partial_experiment_metrics(tmp_path):
    experiment = tmp_path / "experiments/unetpp_effb4_448"
    metrics_dir = experiment / "evaluation_test"
    metrics_dir.mkdir(parents=True)
    (experiment / "failed.json").write_text(json.dumps({"stage": 1}), encoding="utf-8")
    (metrics_dir / "metrics.csv").write_text(
        "split,dice,iou,precision,recall,boundary_f1\n"
        "test,0.86,0.76,0.91,0.82,0.40\n",
        encoding="utf-8",
    )

    outputs = MODULE.write_summary(tmp_path / "experiments", tmp_path / "comparison")

    with outputs["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["experiment"] == "unetpp_effb4_448"
    assert rows[0]["status"] == "partial"
    assert rows[0]["stage"] == "1"
