import json

from notebooks.kaggle_v1_6 import resolve_runtime_roots


def test_v16_kaggle_runtime_roots_from_cloned_repository(monkeypatch, tmp_path):
    repository = tmp_path / "kaggle/working/medical-image-segmentation"
    inputs = tmp_path / "kaggle/input"
    repository.mkdir(parents=True)
    inputs.mkdir()
    monkeypatch.chdir(repository)
    resolved_input, resolved_working = resolve_runtime_roots()
    assert resolved_input == inputs
    assert resolved_working == tmp_path / "kaggle/working"


def test_v16_wrapper_does_not_contain_wandb_runtime_calls():
    source = open("notebooks/kaggle_v1_6.py", encoding="utf-8").read()
    assert "kaggle_secrets" not in source
    assert "wandb.init" not in source
    assert "wandb sync" not in source


def test_v16_kernel_mounts_ham_images_and_reviewed_masks():
    metadata = json.loads(open("kaggle_v1_6_kernel/kernel-metadata.json", encoding="utf-8").read())
    assert "kmader/skin-cancer-mnist-ham10000" in metadata["dataset_sources"]
    assert "tschandl/ham10000-lesion-segmentations" in metadata["dataset_sources"]


def test_v16_debug_mode_runs_preflight_instead_of_long_pipeline():
    source = open("notebooks/kaggle_v1_6.py", encoding="utf-8").read()
    assert "scripts/debug_v1_6.py" in source
    assert "v1_6_debug_report.json" in source


def test_v16_wrapper_forwards_explicit_teacher_reset():
    source = open("notebooks/kaggle_v1_6.py", encoding="utf-8").read()
    assert 'parser.add_argument("--reset-teachers"' in source
    assert 'command.append("--reset-teachers")' in source
