import sys
from types import SimpleNamespace

import pytest

from src.tracking import NullTracker, WandbTracker, _sanitize_config, create_tracker
from src.trainer import _validate_wandb_resume_mapping


class FakeRun:
    def __init__(self, run_id, fail_log=False):
        self.id = run_id
        self.url = f"https://wandb.invalid/{run_id}"
        self.summary = {}
        self.fail_log = fail_log
        self.logged = []

    def log(self, values, step=None):
        if self.fail_log:
            self.fail_log = False
            raise ConnectionError("offline")
        self.logged.append((values, step))

    def log_artifact(self, artifact, aliases=None):
        self.artifact = (artifact, aliases)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class FakeArtifact:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self, fail_online=False, fail_log=False):
        self.calls = []
        self.fail_online = fail_online
        self.fail_log = fail_log

    def init(self, mode, **kwargs):
        self.calls.append((mode, kwargs))
        if mode == "online" and self.fail_online:
            raise ConnectionError("network")
        return FakeRun(kwargs["id"], fail_log=self.fail_log and mode == "online")


def _config():
    return {
        "experiment_name": "teacher-unetpp-fold0",
        "tracking": {
            "enabled": True,
            "project": "medseg-v1-5",
            "run_id": "teacher-unetpp-fold0",
            "group": "v1.5-deadbeef",
        },
    }


def test_tracking_disabled_and_secret_sanitization(tmp_path):
    assert isinstance(create_tracker({"tracking": {"enabled": False}}, tmp_path), NullTracker)
    sanitized = _sanitize_config({"WANDB_API_KEY": "secret", "nested": {"token": "secret", "safe": 1}})
    assert "secret" not in repr(sanitized)
    assert sanitized["nested"]["safe"] == 1


def test_wandb_online_uses_stable_resume_id(monkeypatch, tmp_path):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "not-logged")
    tracker = WandbTracker(_config(), tmp_path)
    assert tracker.run_id == "teacher-unetpp-fold0"
    assert fake.calls[0][0] == "online"
    assert fake.calls[0][1]["resume"] == "allow"
    assert "not-logged" not in repr(fake.calls[0])


def test_wandb_missing_key_or_network_failure_falls_back_offline(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    no_key = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", no_key)
    WandbTracker(_config(), tmp_path / "no-key")
    assert no_key.calls[0][0] == "offline"

    monkeypatch.setenv("WANDB_API_KEY", "value")
    failure = FakeWandb(fail_online=True)
    monkeypatch.setitem(sys.modules, "wandb", failure)
    with pytest.warns(UserWarning, match="switching to offline"):
        tracker = WandbTracker(_config(), tmp_path / "failure")
    assert [call[0] for call in failure.calls] == ["online", "offline"]
    assert tracker.run_id == "teacher-unetpp-fold0"


def test_wandb_required_online_rejects_missing_key_or_network_failure(monkeypatch, tmp_path):
    config = _config()
    config["tracking"]["require_online"] = True
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    missing_key = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", missing_key)
    with pytest.raises(RuntimeError, match="online initialization is required"):
        WandbTracker(config, tmp_path / "missing-key")
    assert missing_key.calls == []

    monkeypatch.setenv("WANDB_API_KEY", "value")
    failure = FakeWandb(fail_online=True)
    monkeypatch.setitem(sys.modules, "wandb", failure)
    with pytest.raises(RuntimeError, match="online initialization is required"):
        WandbTracker(config, tmp_path / "failure")
    assert [call[0] for call in failure.calls] == ["online"]


def test_wandb_log_failure_reopens_same_run_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "value")
    fake = FakeWandb(fail_log=True)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    tracker = WandbTracker(_config(), tmp_path)
    with pytest.warns(UserWarning, match="switching this process to offline"):
        tracker.log({"train/loss": 1.0}, step=1)
    assert [call[0] for call in fake.calls] == ["online", "offline"]
    assert all(call[1]["id"] == "teacher-unetpp-fold0" for call in fake.calls)


def test_checkpoint_resume_requires_same_or_declared_wandb_run():
    checkpoint = {"metadata": {"wandb_run_id": "screen-unetpp-fold0"}}
    tracker = SimpleNamespace(run_id="teacher-unetpp-fold0")
    with pytest.raises(ValueError, match="run mapping mismatch"):
        _validate_wandb_resume_mapping(checkpoint, tracker, {})
    _validate_wandb_resume_mapping(
        checkpoint,
        tracker,
        {"resume_source_run_ids": ["screen-unetpp-fold0"]},
    )
