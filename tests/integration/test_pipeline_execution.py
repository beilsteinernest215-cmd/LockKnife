from __future__ import annotations

import dataclasses
import io
import json
import pathlib
import tarfile

import pytest

from lockknife.core.case import load_case_manifest
from lockknife.core.pipeline import (
    PipelineExecutor,
    PlaybookDefinition,
    StepDefinition,
    get_playbook,
    list_checkpoints,
)
from lockknife_headless_cli.tui_callback import build_tui_callback


def test_end_to_end_pipeline_execution_in_case_workspace(tmp_path: pathlib.Path) -> None:
    case_dir = tmp_path / "CASE-E2E"

    step1 = StepDefinition(
        step_id="report.stage1",
        label="Pre-Report Initialization",
        action_id="report.generate",
        category="reporting",
        params={"template": "technical", "format": "json"},
    )
    step2 = StepDefinition(
        step_id="report.stage2",
        label="Final Report Synthesis",
        action_id="report.generate",
        category="reporting",
        depends_on=("report.stage1",),
        params={"template": "technical", "format": "json"},
    )

    playbook = PlaybookDefinition(
        name="e2e-pipeline",
        title="E2E Pipeline Test",
        description="Verifies full pipeline execution against real case workspace.",
        category="integration",
        steps=(step1, step2),
    )

    executor = PipelineExecutor(case_dir=case_dir, concurrency=2)
    summary = executor.execute(playbook)

    assert summary.status.value in {"completed", "partial"}
    assert summary.total_steps == 2
    assert summary.completed_steps == 2

    # Verify checkpoint on disk
    checkpoints = list_checkpoints(case_dir)
    assert len(checkpoints) == 1
    cp = checkpoints[0]
    assert cp.pipeline_id == summary.pipeline_id
    assert cp.playbook_name == "e2e-pipeline"
    assert "report.stage1" in cp.step_records
    assert "report.stage2" in cp.step_records

    # Verify manifest exists and is valid
    manifest = load_case_manifest(case_dir)
    assert manifest.case_id == "CASE-E2E"


@pytest.mark.parametrize(
    "name", ["triage", "deep-forensics", "incident-response", "crypto-audit", "full-spectrum"]
)
def test_builtins_use_real_offline_handlers_and_acquired_files(
    tmp_path: pathlib.Path, name: str
) -> None:
    """Stub device access only; dispatch analysis and reporting through the real registry."""
    callback = build_tui_callback(None)
    pb = get_playbook(name)
    device_actions = {step.action_id for step in pb.steps if step.requires_device}

    def dispatch(action: str, params: dict) -> dict:
        if action in device_actions:
            if params.get("output"):
                output = pathlib.Path(params["output"])
                if action == "forensics.snapshot":
                    output.write_bytes(b"snapshot bytes %PDF-1.4 sample %%EOF")
                else:
                    output.write_text(
                        json.dumps(
                            [
                                {
                                    "date_ms": 1700000000000,
                                    "timestamp_ms": 1700000000000,
                                    "body": "pipeline fixture",
                                    "address": "+15555550123",
                                }
                            ]
                        ),
                        encoding="utf-8",
                    )
            return {"ok": True}
        return callback(action, params)

    summary = PipelineExecutor(
        case_dir=tmp_path / name, target_serial="FIXTURE", dispatch=dispatch
    ).execute(pb)
    assert summary.status.value == "completed", [
        (r.step_id, r.error_message) for r in summary.step_records
    ]
    records = {record.step_id: record for record in summary.step_records}
    if "forensics.timeline" in records:
        timeline = json.loads(
            pathlib.Path(records["forensics.timeline"].summary["output"]).read_text()
        )
        assert timeline["event_count"] > 0
        assert any("sms.json" in source["path"] for source in timeline["sources"])
        carved = records["forensics.carve"].summary["data"]
        assert carved["carved_count"] == 1
        assert pathlib.Path(carved["carved"][0]["path"]).is_file()


@pytest.mark.parametrize("encrypt,full", [(False, True), (False, False), (True, True)])
def test_production_snapshot_pipeline_with_stub_device(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    encrypt: bool,
    full: bool,
) -> None:
    commands = []

    class Device:
        def has_root(self, serial: str) -> bool:
            return True

        def shell(self, serial: str, command: str, **kwargs: object) -> str:
            commands.append(command)
            return ""

        def pull(self, serial: str, remote: str, local: pathlib.Path, **kwargs: object) -> None:
            with tarfile.open(local, "w") as archive:
                content = b"%PDF-1.4 sample %%EOF"
                entry = tarfile.TarInfo("evidence.pdf")
                entry.size = len(content)
                archive.addfile(entry, io.BytesIO(content))

    monkeypatch.setattr("lockknife_headless_cli.tui_callback._devices", lambda app: Device())
    callback = build_tui_callback(None)
    builtin = next(
        s for s in get_playbook("deep-forensics").steps if s.action_id == "forensics.snapshot"
    )
    assert builtin.params["full"] is True
    snapshot = dataclasses.replace(
        builtin,
        depends_on=(),
        params={**builtin.params, "encrypt": encrypt, "full": full, "paths": "/data/example"},
    )
    steps = [snapshot]
    if not encrypt:
        steps.append(
            StepDefinition(
                "carve",
                "Carve Snapshot",
                "forensics.carve",
                "forensics",
                depends_on=(snapshot.step_id,),
                params={
                    "path": {"$result": snapshot.step_id, "path": "output"},
                    "output_dir": "${run_dir}/carved",
                },
            )
        )
    pb = PlaybookDefinition("snapshot-stub", "Snapshot", "", "test", tuple(steps))
    summary = PipelineExecutor(
        case_dir=tmp_path, target_serial="FIXTURE", dispatch=callback
    ).execute(pb)
    assert summary.status.value == "completed", summary.to_dict()
    output = pathlib.Path(summary.step_records[0].summary["output"])
    assert output.is_file()
    assert output.name.endswith(".lkenc") is encrypt
    assert any(a.path.endswith(output.name) for a in load_case_manifest(tmp_path).artifacts)
    selected = ("/data/system", "/data/data") if full else ("/data/example",)
    assert any(all(path in command for path in selected) for command in commands)
    if not encrypt:
        assert summary.step_records[1].summary["data"]["carved_count"] == 1
