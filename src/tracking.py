from __future__ import annotations

import json
import os
import warnings
from pathlib import Path


class NullTracker:
    run_id = None
    run_url = None

    def log(self, metrics, step=None):
        del metrics, step

    def summary(self, values):
        del values

    def log_histogram(self, name, values, step=None):
        del name, values, step

    def log_table(self, name, columns, rows, step=None):
        del name, columns, rows, step

    def log_artifact(self, name, files, artifact_type="model", aliases=None, metadata=None):
        del name, files, artifact_type, aliases, metadata

    def finish(self, exit_code=0):
        del exit_code


class WandbTracker(NullTracker):
    def __init__(self, config, output_dir):
        tracking = config.get("tracking", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(tracking.get("run_id") or config.get("experiment_name", "run"))[:64]
        self.run_url = None
        self._wandb = None
        self._run = None
        self._init_kwargs = {}
        try:
            import wandb

            self._wandb = wandb
            requested_mode = str(tracking.get("mode", "online"))
            if requested_mode == "online" and not os.environ.get("WANDB_API_KEY"):
                requested_mode = "offline"
            default_offline = Path(os.environ.get("WANDB_DIR", self.output_dir / "wandb-offline"))
            wandb_dir = Path(tracking.get("offline_dir", default_offline))
            wandb_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("WANDB_DIR", str(wandb_dir))
            self._init_kwargs = {
                "project": str(tracking.get("project", "medseg-v1-5")),
                "entity": tracking.get("entity") or os.environ.get("WANDB_ENTITY"),
                "group": tracking.get("group"),
                "id": self.run_id,
                "name": str(tracking.get("name") or self.run_id),
                "job_type": tracking.get("job_type", "training"),
                "tags": list(tracking.get("tags", [])),
                "config": _sanitize_config(config),
                "resume": "allow",
                "dir": str(wandb_dir),
            }
            self._run = wandb.init(mode=requested_mode, **self._init_kwargs)
            self.run_url = getattr(self._run, "url", None)
            self._write_state(requested_mode)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"W&B online initialization failed; switching to offline logging: {exc}", stacklevel=2)
            self._start_offline()

    def _start_offline(self):
        if self._wandb is None or not self._init_kwargs:
            self._run = None
            return
        try:
            self._run = self._wandb.init(mode="offline", **self._init_kwargs)
            self.run_url = getattr(self._run, "url", None)
            self._write_state("offline")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"W&B offline initialization failed; training continues with local CSV logs: {exc}", stacklevel=2
            )
            self._run = None

    def _write_state(self, mode):
        state = {"run_id": self.run_id, "run_url": self.run_url, "mode": mode}
        (self.output_dir / "wandb_run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def log(self, metrics, step=None):
        if self._run is None:
            return
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"W&B logging failed; switching this process to offline logging: {exc}", stacklevel=2)
            try:
                self._run.finish(exit_code=1)
            except Exception:  # noqa: BLE001
                pass
            self._start_offline()
            if self._run is not None:
                try:
                    self._run.log(dict(metrics), step=step)
                except Exception as offline_exc:  # noqa: BLE001
                    warnings.warn(
                        f"W&B offline logging failed; local CSV logs remain available: {offline_exc}", stacklevel=2
                    )

    def summary(self, values):
        if self._run is None:
            return
        try:
            self._run.summary.update(dict(values))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"W&B summary update failed: {exc}", stacklevel=2)

    def log_histogram(self, name, values, step=None):
        if self._run is None or self._wandb is None:
            return
        try:
            self._run.log({name: self._wandb.Histogram(list(values))}, step=step)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"W&B histogram logging failed: {exc}", stacklevel=2)

    def log_table(self, name, columns, rows, step=None):
        if self._run is None or self._wandb is None:
            return
        try:
            table = self._wandb.Table(columns=list(columns), data=[list(row) for row in rows])
            self._run.log({name: table}, step=step)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"W&B table logging failed: {exc}", stacklevel=2)

    def log_artifact(self, name, files, artifact_type="model", aliases=None, metadata=None):
        if self._run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
            for path in files:
                path = Path(path)
                if path.exists() and path.is_file():
                    artifact.add_file(str(path), name=path.name)
            self._run.log_artifact(artifact, aliases=list(aliases or []))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"W&B artifact upload failed: {exc}", stacklevel=2)

    def finish(self, exit_code=0):
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"W&B finish failed: {exc}", stacklevel=2)


def _sanitize_config(config):
    secret_markers = {"api_key", "token", "password", "secret"}

    def clean(value):
        if isinstance(value, dict):
            return {
                key: (
                    "<redacted>"
                    if any(marker in str(key).lower() for marker in secret_markers)
                    else clean(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return clean(config)


def create_tracker(config, output_dir):
    tracking = config.get("tracking", {})
    if not bool(tracking.get("enabled", False)):
        return NullTracker()
    return WandbTracker(config, output_dir)
