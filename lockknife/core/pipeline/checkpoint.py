from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile

from lockknife.core.exceptions import LockKnifeError
from lockknife.core.pipeline.models import PipelineCheckpoint


def pipeline_dir(case_dir: pathlib.Path) -> pathlib.Path:
    return case_dir / "pipelines"


def pipeline_checkpoint_path(case_dir: pathlib.Path, pipeline_id: str) -> pathlib.Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", pipeline_id):
        raise LockKnifeError(
            "Invalid pipeline ID: use 1-128 letters, digits, underscores or hyphens"
        )
    directory = pipeline_dir(case_dir)
    path = directory / f"{pipeline_id}.json"
    if directory.is_symlink() or path.is_symlink():
        raise LockKnifeError("Pipeline checkpoint paths must not be symlinks")
    return path


def save_checkpoint(checkpoint: PipelineCheckpoint) -> pathlib.Path:
    path = pipeline_checkpoint_path(pathlib.Path(checkpoint.case_dir), checkpoint.pipeline_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".checkpoint-", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(checkpoint.to_dict(), handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            pathlib.Path(temporary).unlink(missing_ok=True)
    return path


def load_checkpoint(case_dir: pathlib.Path, pipeline_id: str) -> PipelineCheckpoint:
    path = pipeline_checkpoint_path(case_dir, pipeline_id)
    if not path.exists():
        raise LockKnifeError(f"Pipeline checkpoint not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Expected dictionary payload")
        checkpoint = PipelineCheckpoint.from_dict(payload)
        if (
            checkpoint.pipeline_id != pipeline_id
            or pathlib.Path(checkpoint.case_dir).resolve() != case_dir.resolve()
        ):
            raise ValueError("Checkpoint identity does not match its path")
        return checkpoint
    except Exception as exc:
        raise LockKnifeError(f"Failed to load pipeline checkpoint {path}: {exc}") from exc


def list_checkpoints(case_dir: pathlib.Path) -> list[PipelineCheckpoint]:
    p_dir = case_dir / "pipelines"
    if not p_dir.exists():
        return []

    checkpoints: list[PipelineCheckpoint] = []
    for f in sorted(p_dir.glob("*.json")):
        try:
            checkpoints.append(load_checkpoint(case_dir, f.stem))
        except Exception:
            continue

    # Sort descending by updated_at_utc or started_at_utc
    checkpoints.sort(key=lambda cp: cp.updated_at_utc, reverse=True)
    return checkpoints


def find_latest_checkpoint(
    case_dir: pathlib.Path, playbook_name: str | None = None
) -> PipelineCheckpoint | None:
    cps = list_checkpoints(case_dir)
    if playbook_name:
        cps = [cp for cp in cps if cp.playbook_name == playbook_name]
    return cps[0] if cps else None
