from pathlib import Path

from notebooks.kaggle_v1_5 import resolve_runtime_roots


def test_resolve_runtime_roots_from_cloned_kaggle_repository(monkeypatch, tmp_path):
    kaggle_root = tmp_path / "kaggle"
    repository = kaggle_root / "working/medical-image-segmentation"
    input_root = kaggle_root / "input"
    repository.mkdir(parents=True)
    input_root.mkdir()
    monkeypatch.chdir(repository)

    resolved_input, resolved_working = resolve_runtime_roots()

    assert resolved_input == input_root
    assert resolved_working == kaggle_root / "working"


def test_resolve_runtime_roots_prefers_explicit_arguments(tmp_path):
    input_root = tmp_path / "mounted-input"
    working_root = tmp_path / "scratch"

    resolved_input, resolved_working = resolve_runtime_roots(input_root, working_root)

    assert resolved_input == Path(input_root).resolve()
    assert resolved_working == Path(working_root).resolve()
