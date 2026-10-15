from __future__ import annotations

import concurrent.futures
import hashlib
import json
import pathlib
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from lockknife.core._case_common import _utc_now
from lockknife.core._case_store import is_case_workspace
from lockknife.core.case import create_case_workspace, load_case_manifest
from lockknife.core.exceptions import LockKnifeError
from lockknife.core.pipeline.checkpoint import (
    list_checkpoints,
    load_checkpoint,
    pipeline_checkpoint_path,
    save_checkpoint,
)
from lockknife.core.pipeline.dag import build_dag, resolve_execution_tiers
from lockknife.core.pipeline.models import (
    AcquisitionTier,
    PipelineCheckpoint,
    PipelineStatus,
    PipelineSummary,
    PlaybookDefinition,
    StepDefinition,
    StepExecutionRecord,
    StepStatus,
)
from lockknife.core.pipeline.validation import validate_action, validate_recipe


def _normalize_result(result: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    from lockknife_headless_cli._tui_callback_helpers import _payload_has_partial_signal

    result = dict(result)
    for key in ("data", "job"):
        encoded = result.get(f"{key}_json")
        if encoded is not None:
            try:
                result[key] = json.loads(encoded)
            except (TypeError, ValueError):
                result.update(ok=False, error=f"Invalid {key}_json action result")
    for key in ("data", "job"):
        payload = result.get(key)
        if isinstance(payload, dict):
            if (payload.get("success") is False or payload.get("ok") is False
                    or payload.get("status") in {"failed", "aborted", "cancelled"}):
                result.update(ok=False, error=payload.get("error") or payload.get("error_message")
                              or f"Action {key} reports failure")
            elif payload.get("status") == "partial" or _payload_has_partial_signal(payload):
                result["partial"] = True
        elif _payload_has_partial_signal(payload):
            result["partial"] = True
    if result.get("success") is False or result.get("status") in {"failed", "aborted", "cancelled"}:
        result.update(ok=False, error=result.get("error") or "Action reports failure")
    if result.get("ok") and params.get("output"):
        payload = result.get("data")
        reported_output = result.get("output") or (
            payload.get("output") if isinstance(payload, dict) else None
        )
        output = pathlib.Path(reported_output or params["output"])
        if not output.is_file():
            result.update(ok=False, error=f"Declared output file was not created: {output}")
        else:
            result["output"] = str(output)
    return result


class PipelineExecutor:
    """Execute DAG tiers; cancellation and deadlines never abandon running handlers.

    Handlers are synchronous and cannot be forcibly interrupted safely. Deadlines
    are checked after each call returns; cancellation stops new calls and waits
    for in-flight calls. Timed-out calls are never retried or sent to a fallback.
    """

    def __init__(
        self, *, case_dir: pathlib.Path, target_serial: str | None = None,
        concurrency: int = 4, stop_on_failure: bool = False,
        dispatch: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        action_validator: Callable[[str, dict[str, Any]], None] = validate_action,
    ) -> None:
        self.case_dir = pathlib.Path(case_dir).resolve()
        self.target_serial = target_serial
        self.concurrency = max(1, concurrency)
        self.stop_on_failure = stop_on_failure
        self._action_callback = dispatch
        self._action_validator = action_validator
        self._device_lock = threading.Lock()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Stop scheduling work; active synchronous handlers must return first."""
        self._cancel.set()

    def _get_action_callback(self) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
        if self._action_callback is None:
            from lockknife_headless_cli.tui_callback import build_tui_callback

            self._action_callback = build_tui_callback(None)
        return self._action_callback

    def _params(self, step: StepDefinition, run_dir: pathlib.Path,
                records: dict[str, StepExecutionRecord] | None = None) -> dict[str, Any]:
        def resolve(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace("${case_dir}", str(self.case_dir)).replace("${run_dir}", str(run_dir))
            if isinstance(value, dict):
                if "$result" in value:
                    if records is None:
                        return value
                    record = records.get(value["$result"])
                    if record is None or record.status != StepStatus.COMPLETED:
                        raise LockKnifeError(f"Result unavailable: {value['$result']}")
                    result: Any = record.summary
                    for key in value["path"].split("."):
                        if not isinstance(result, dict) or key not in result:
                            raise LockKnifeError(f"Missing result field: {value}")
                        result = result[key]
                    if value["path"] == "output" and not pathlib.Path(result).is_file():
                        raise LockKnifeError(f"Result output file is missing: {result}")
                    return result
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [resolve(item) for item in value]
            return value

        params = resolve(step.params)
        params["case_dir"] = str(self.case_dir)
        if self.target_serial:
            params["serial"] = self.target_serial
            params["device_id"] = self.target_serial
        return params

    def plan(self, playbook: PlaybookDefinition) -> dict[str, Any]:
        dag = build_dag(playbook.steps)
        validate_recipe(playbook)
        tiers = resolve_execution_tiers(dag)
        for step in playbook.steps:
            params = self._params(step, self.case_dir / "pipelines" / "PREVIEW")
            # A plan can be inspected offline, but live dispatch requires a serial.
            if step.requires_device and not self.target_serial:
                params["serial"] = "<target-serial>"
            self._action_validator(step.action_id, params)
            if step.fallback_action_id:
                self._action_validator(step.fallback_action_id, params)
        return {
            "playbook_name": playbook.name, "title": playbook.title,
            "description": playbook.description, "category": playbook.category,
            "case_dir": str(self.case_dir), "target_serial": self.target_serial,
            "requires_device": any(s.requires_device for s in playbook.steps),
            "total_steps": len(playbook.steps), "tier_count": len(tiers),
            "concurrency": self.concurrency,
            "timeout_policy": "deadline checked after handler returns; no forced interruption or timeout retry",
            "cancellation_policy": "stop new calls and wait for active handlers",
            "stop_on_failure_policy": "finish current tier; skip subsequent tiers after a required failure",
            "tiers": [[s.to_dict() for s in tier] for tier in tiers],
        }

    def execute(
        self, playbook: PlaybookDefinition, *, resume: bool = False,
        dry_run: bool = False, pipeline_id: str | None = None,
        on_step_start: Callable[[StepDefinition], None] | None = None,
        on_step_complete: Callable[[StepDefinition, StepExecutionRecord], None] | None = None,
    ) -> PipelineSummary:
        start = time.perf_counter()
        self.plan(playbook)
        fingerprint = hashlib.sha256(json.dumps(playbook.to_dict(), sort_keys=True,
                                               allow_nan=False).encode()).hexdigest()
        metadata = {"title": playbook.title, "recipe": playbook.to_dict(),
                    "recipe_fingerprint": fingerprint, "dry_run": False,
                    "stop_on_failure": self.stop_on_failure}
        active_id = pipeline_id if pipeline_id is not None else f"pipe-{uuid.uuid4().hex}"
        pipeline_checkpoint_path(self.case_dir, active_id)
        checkpoint = None
        if resume and not dry_run:
            candidates = ([load_checkpoint(self.case_dir, pipeline_id)] if pipeline_id else
                          list_checkpoints(self.case_dir))
            for candidate in candidates:
                matches = (
                    candidate.playbook_name == playbook.name
                    and candidate.target_serial == self.target_serial
                    and candidate.metadata.get("recipe_fingerprint") == fingerprint
                    and candidate.metadata.get("dry_run") is False
                    and candidate.metadata.get("stop_on_failure") == self.stop_on_failure
                )
                if matches:
                    checkpoint = candidate
                    active_id = candidate.pipeline_id
                    break
                if pipeline_id:
                    raise LockKnifeError("Checkpoint recipe, target or execution mode mismatch")
        elif pipeline_id and not dry_run and pipeline_checkpoint_path(self.case_dir, active_id).exists():
            raise LockKnifeError("Pipeline ID already exists; use resume or a new ID")

        records = dict(checkpoint.step_records) if checkpoint else {}
        known_ids = {step.step_id for step in playbook.steps}
        if set(records) - known_ids:
            raise LockKnifeError("Checkpoint contains steps outside the recipe")
        if any(key != record.step_id for key, record in records.items()):
            raise LockKnifeError("Checkpoint step identity mismatch")
        started_at = checkpoint.started_at_utc if checkpoint else _utc_now()
        run_dir = self.case_dir / "pipelines" / active_id

        if dry_run:
            previews = tuple(StepExecutionRecord(
                step_id=s.step_id, status=StepStatus.PENDING, attempt_count=0,
                summary={"dry_run": True, "action_id": s.action_id},
            ) for s in playbook.steps)
            return PipelineSummary(active_id, playbook.name, str(self.case_dir), self.target_serial,
                                   PipelineStatus.PENDING, time.perf_counter() - start,
                                   len(previews), 0, 0, 0, 0, previews, "")

        if not is_case_workspace(self.case_dir):
            create_case_workspace(case_dir=self.case_dir, case_id=self.case_dir.name or "CASE-AUTO",
                                  examiner="LockKnife Pipeline", title=playbook.title)
        run_dir.mkdir(parents=True, exist_ok=True)

        def persist(status: PipelineStatus, *, ended: bool = False) -> pathlib.Path:
            now = _utc_now()
            return save_checkpoint(PipelineCheckpoint(
                active_id, playbook.name, str(self.case_dir), self.target_serial, status,
                started_at, now, now if ended else None, dict(records), metadata,
            ))

        persist(PipelineStatus.RUNNING)
        dispatch = self._get_action_callback()
        stopped = False

        def run_step(step: StepDefinition) -> StepExecutionRecord:
            step_start = time.perf_counter()
            now = _utc_now()
            attempts = 0
            fallback = False
            result: dict[str, Any] = {}
            try:
                if self._cancel.is_set():
                    return StepExecutionRecord(step.step_id, StepStatus.SKIPPED, attempt_count=0,
                                               error_message="Cancelled before dispatch")
                if on_step_start:
                    on_step_start(step)
                params = self._params(step, run_dir, records)
                for key in ("output", "output_dir"):
                    if isinstance(params.get(key), str):
                        path = pathlib.Path(params[key])
                        (path if key == "output_dir" else path.parent).mkdir(parents=True, exist_ok=True)
                actions = [step.action_id] + ([step.fallback_action_id] if step.fallback_action_id else [])
                if step.requires_device and not self.target_serial:
                    if step.fallback_action_id:
                        actions = [step.fallback_action_id]
                    else:
                        raise LockKnifeError(f"Step {step.step_id} requires a target device serial")
                deadline = step_start + step.timeout_s
                expired = False
                for action in actions:
                    fallback = action != step.action_id
                    for retry in range(step.retries + 1):
                        if self._cancel.is_set() or time.perf_counter() >= deadline:
                            expired = time.perf_counter() >= deadline
                            result = {"ok": False, "error": "Deadline exceeded" if expired else "Cancelled"}
                            break
                        try:
                            self._action_validator(action, params)
                            # The production callback uses shared job tracking state.
                            # Serialize it as well as device actions; injected dispatchers
                            # may safely execute independent offline work concurrently.
                            with self._device_lock if step.requires_device or self._action_validator is validate_action else nullcontext():
                                if self._cancel.is_set() or time.perf_counter() >= deadline:
                                    result = {"ok": False, "error": "Cancelled or deadline exceeded before dispatch"}
                                    break
                                attempts += 1
                                result = dispatch(action, dict(params))
                            if not isinstance(result, dict):
                                raise ValueError("Action returned a non-object result")
                            result = _normalize_result(result, params)
                        except Exception as exc:
                            result = {"ok": False, "error": str(exc)}
                        if time.perf_counter() >= deadline:
                            expired = True
                            result = {**result, "ok": False,
                                      "error": "Deadline exceeded; handler has returned (not forcibly interrupted)"}
                            break
                        if result.get("ok") or self._cancel.is_set():
                            break
                        if retry < step.retries:
                            remaining = max(0.0, deadline - time.perf_counter())
                            self._cancel.wait(min(step.retry_backoff_s * 2**retry, remaining))
                    if result.get("ok") or expired or self._cancel.is_set():
                        break
                result = dict(result)
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            status = StepStatus.COMPLETED if result.get("ok") else StepStatus.FAILED
            if result.get("ok") and (result.get("partial") or result.get("status") == "partial"):
                status = StepStatus.PARTIAL
            return StepExecutionRecord(
                step.step_id, status, now, _utc_now(), (time.perf_counter() - step_start) * 1000,
                AcquisitionTier.DIRECT if step.requires_device else AcquisitionTier.LOCAL_DERIVED,
                summary=result, error_message=result.get("error"),
                recovery_hint=result.get("recovery_hint"), attempt_count=attempts, used_fallback=fallback,
            )

        for tier in resolve_execution_tiers(build_dag(playbook.steps)):
            runnable = []
            for step in tier:
                if step.step_id in records and records[step.step_id].status == StepStatus.COMPLETED:
                    continue
                unmet = any(
                    (records.get(dep) is None or records[dep].status != StepStatus.COMPLETED)
                    and not next(s for s in playbook.steps if s.step_id == dep).optional
                    for dep in step.depends_on
                )
                if stopped or self._cancel.is_set() or unmet:
                    records[step.step_id] = StepExecutionRecord(
                        step.step_id, StepStatus.SKIPPED, attempt_count=0,
                        error_message="Cancelled" if self._cancel.is_set() else
                        "Stopped after failure" if stopped else "Prerequisite did not complete",
                    )
                    if on_step_complete:
                        on_step_complete(step, records[step.step_id])
                else:
                    runnable.append(step)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(run_step, step): step for step in runnable}
                for future in concurrent.futures.as_completed(futures):
                    step = futures[future]
                    records[step.step_id] = future.result()
                    persist(PipelineStatus.RUNNING)
                    if on_step_complete:
                        on_step_complete(step, records[step.step_id])
            if self.stop_on_failure and any(
                records[s.step_id].status in {StepStatus.FAILED, StepStatus.PARTIAL} and not s.optional
                for s in tier
            ):
                stopped = True
            persist(PipelineStatus.RUNNING)

        completed = sum(r.status == StepStatus.COMPLETED for r in records.values())
        skipped = sum(r.status == StepStatus.SKIPPED for r in records.values())
        failed = sum(r.status == StepStatus.FAILED for r in records.values())
        partial = any(r.status == StepStatus.PARTIAL for r in records.values())
        status = (PipelineStatus.CANCELLED if self._cancel.is_set() else
                  PipelineStatus.COMPLETED if completed == len(playbook.steps) else
                  PipelineStatus.PARTIAL if completed or partial else PipelineStatus.FAILED)
        path = persist(status, ended=True)
        artifact_count = len(load_case_manifest(self.case_dir).artifacts)
        return PipelineSummary(
            active_id, playbook.name, str(self.case_dir), self.target_serial, status,
            time.perf_counter() - start, len(playbook.steps), completed, skipped, failed,
            artifact_count, tuple(records[s.step_id] for s in playbook.steps), str(path),
        )
