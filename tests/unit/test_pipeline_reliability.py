from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import pytest

from lockknife.core.exceptions import LockKnifeError
from lockknife.core.pipeline import PipelineExecutor, PlaybookDefinition, StepDefinition
from lockknife.core.pipeline.checkpoint import list_checkpoints, load_checkpoint, save_checkpoint
from lockknife.core.pipeline.models import PipelineStatus, StepStatus
from lockknife.core.pipeline.playbooks import list_playbooks
from lockknife_headless_cli.tui_callback import build_tui_callback


def recipe(*steps: StepDefinition) -> PlaybookDefinition:
    return PlaybookDefinition("custom", "Custom", "Test", "test", steps)


def step(name: str = "one", **kwargs: object) -> StepDefinition:
    return StepDefinition(name, name, name, "test", **kwargs)


def executor(path: Path, **kwargs: object) -> PipelineExecutor:
    return PipelineExecutor(case_dir=path, action_validator=lambda action, params: None, **kwargs)


def test_all_builtins_validate_against_real_registry_without_io(tmp_path: Path) -> None:
    engine = PipelineExecutor(case_dir=tmp_path / "case")
    for playbook in list_playbooks():
        assert engine.plan(playbook)["total_steps"] == len(playbook.steps)
    assert not engine.case_dir.exists()


@pytest.mark.parametrize("action,params", [
    ("security.scan", {}), ("forensics.carve", {"path": "x"}),
    ("forensics.parse", {}), ("security.malware", {}),
    ("report.generate", {"format": "invalid"}),
])
def test_invalid_actions_and_params_fail_planning(tmp_path: Path, action: str, params: dict) -> None:
    with pytest.raises(LockKnifeError):
        PipelineExecutor(case_dir=tmp_path).plan(recipe(dataclasses.replace(step(), action_id=action, params=params)))


def test_invalid_reference_and_retry_settings_fail_planning(tmp_path: Path) -> None:
    engine = executor(tmp_path)
    for bad in (step(params={"path": {"$result": "missing", "path": "output"}}),
                step(timeout_s=float("nan")), step(retries=-1), step(retry_backoff_s=-1)):
        with pytest.raises(LockKnifeError):
            engine.plan(recipe(bad))


def test_dry_run_cannot_poison_resume(tmp_path: Path) -> None:
    calls = []
    engine = executor(tmp_path / "case", dispatch=lambda action, params: calls.append(action) or {"ok": True})
    pb = recipe(step())
    preview = engine.execute(pb, dry_run=True, pipeline_id="same")
    assert preview.status == PipelineStatus.PENDING
    assert list_checkpoints(engine.case_dir) == []
    assert not engine.case_dir.exists()
    live = engine.execute(pb, resume=True)
    assert live.completed_steps == 1
    assert calls == ["one"]
    engine.execute(pb, dry_run=True, resume=True, pipeline_id=live.pipeline_id)
    assert load_checkpoint(engine.case_dir, live.pipeline_id).status == PipelineStatus.COMPLETED


@pytest.mark.parametrize("change", ["recipe", "target", "mode", "dry_run"])
def test_explicit_resume_rejects_mismatch(tmp_path: Path, change: str) -> None:
    pb = recipe(step())
    engine = executor(tmp_path, target_serial="A", dispatch=lambda *_: {"ok": True})
    first = engine.execute(pb)
    if change == "recipe":
        pb = recipe(step(params={"new": True}))
    elif change == "target":
        engine.target_serial = "B"
    elif change == "mode":
        engine.stop_on_failure = True
    else:
        cp = load_checkpoint(tmp_path, first.pipeline_id)
        save_checkpoint(dataclasses.replace(cp, metadata={**cp.metadata, "dry_run": True}))
    with pytest.raises(LockKnifeError, match="mismatch"):
        engine.execute(pb, resume=True, pipeline_id=first.pipeline_id)


def test_explicit_missing_resume_does_not_fall_back(tmp_path: Path) -> None:
    engine = executor(tmp_path, dispatch=lambda *_: {"ok": True})
    engine.execute(recipe(step()))
    with pytest.raises(LockKnifeError, match="not found"):
        engine.execute(recipe(step()), resume=True, pipeline_id="absent")


@pytest.mark.parametrize("bad", ["../outside", "/tmp/outside", "a/b", "..", "", "a\\b"])
def test_checkpoint_id_validation(tmp_path: Path, bad: str) -> None:
    with pytest.raises(LockKnifeError, match="Invalid pipeline ID"):
        load_checkpoint(tmp_path / "case", bad)
    assert not (tmp_path / "case").exists()


def test_atomic_failure_preserves_previous_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = executor(tmp_path, dispatch=lambda *_: {"ok": True})
    result = engine.execute(recipe(step()))
    cp = load_checkpoint(tmp_path, result.pipeline_id)
    original = Path(result.checkpoint_path).read_bytes()
    def fail(*args: object) -> None:
        raise OSError("replace failed")
    monkeypatch.setattr("lockknife.core.pipeline.checkpoint.os.replace", fail)
    with pytest.raises(OSError):
        save_checkpoint(dataclasses.replace(cp, status=PipelineStatus.FAILED))
    assert Path(result.checkpoint_path).read_bytes() == original
    assert not list((tmp_path / "pipelines").glob(".checkpoint-*"))


def test_retry_attempt_count_and_stop_after_tier(tmp_path: Path) -> None:
    calls = []
    engine = executor(tmp_path, stop_on_failure=True,
                      dispatch=lambda action, params: calls.append(action) or {"ok": False})
    result = engine.execute(recipe(step(retries=2, retry_backoff_s=0), step("two", depends_on=("one",))))
    assert calls == ["one"] * 3
    assert result.step_records[0].attempt_count == 3
    assert result.step_records[1].status == StepStatus.SKIPPED
    assert result.status == PipelineStatus.FAILED


def test_timeout_waits_for_handler_and_never_retries(tmp_path: Path) -> None:
    finished = threading.Event()
    calls = []
    def dispatch(action: str, params: dict) -> dict:
        calls.append(action)
        time.sleep(0.04)
        finished.set()
        return {"ok": True}
    engine = executor(tmp_path, dispatch=dispatch)
    result = engine.execute(recipe(step(timeout_s=0.01, retries=3, fallback_action_id="fallback")))
    assert finished.is_set()
    assert calls == ["one"]
    assert result.failed_steps == 1
    assert "not forcibly interrupted" in result.step_records[0].error_message


def test_cancel_stops_following_tiers_without_abandoning_work(tmp_path: Path) -> None:
    calls = []
    engine = executor(tmp_path)
    def dispatch(action: str, params: dict) -> dict:
        calls.append(action)
        engine.cancel()
        return {"ok": True}
    engine._action_callback = dispatch
    result = engine.execute(recipe(step(), step("two", depends_on=("one",))))
    assert result.status == PipelineStatus.CANCELLED
    assert calls == ["one"]
    assert result.step_records[1].status == StepStatus.SKIPPED


@pytest.mark.parametrize("response,expected", [
    ({"ok": False}, PipelineStatus.FAILED),
    ({"ok": True, "partial": True}, PipelineStatus.PARTIAL),
])
def test_optional_and_partial_results_are_not_completed(tmp_path: Path, response: dict, expected: PipelineStatus) -> None:
    result = executor(tmp_path, dispatch=lambda *_: response).execute(recipe(step(optional=True)))
    assert result.status == expected
    assert result.completed_steps == 0


def test_missing_result_file_blocks_downstream(tmp_path: Path) -> None:
    calls = []
    result = executor(tmp_path, dispatch=lambda action, params: calls.append(action) or {"ok": True}).execute(
        recipe(step(params={"output": "${run_dir}/missing.json"}),
               step("two", depends_on=("one",), params={"path": {"$result": "one", "path": "output"}})))
    assert calls == ["one"]
    assert result.failed_steps == 1
    assert "not created" in result.step_records[0].error_message
    assert result.step_records[1].status == StepStatus.SKIPPED


@pytest.mark.parametrize("result,expected", [
    ({"ok": True, "data_json": '{"success": false}'}, StepStatus.FAILED),
    ({"ok": True, "data_json": '{"status": "partial"}'}, StepStatus.PARTIAL),
    ({"ok": True, "data_json": '{"warnings": ["incomplete"]}'}, StepStatus.PARTIAL),
    ({"ok": True, "job_json": '{"status": "partial"}'}, StepStatus.PARTIAL),
    ({"ok": True, "job_json": '{"status": "failed"}'}, StepStatus.FAILED),
    ({"ok": True, "data_json": 'invalid'}, StepStatus.FAILED),
])
def test_bridge_payload_and_job_outcomes(tmp_path: Path, result: dict, expected: StepStatus) -> None:
    summary = executor(tmp_path, dispatch=lambda *_: result).execute(recipe(step()))
    assert summary.step_records[0].status == expected
    assert summary.completed_steps == 0


def test_tui_run_supports_stop_on_failure_and_dry_run(tmp_path: Path) -> None:
    result = build_tui_callback(None)("pipeline.run", {
        "case_dir": str(tmp_path / "case"), "stop_on_failure": True, "dry_run": True,
    })
    assert result["ok"] is True
    assert json.loads(result["data_json"])["status"] == "pending"
    assert not (tmp_path / "case").exists()
