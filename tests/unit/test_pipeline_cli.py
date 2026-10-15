from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from lockknife_headless_cli.main import cli


@pytest.mark.parametrize("out_format", ["text", "json"])
@pytest.mark.parametrize(
    "status,code",
    [("failed", 1), ("partial", 1), ("cancelled", 1), ("completed", 0), ("pending", 0)],
)
def test_run_exit_status_after_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, out_format: str, status: str, code: int
) -> None:
    from lockknife.core.pipeline import PipelineExecutor

    summary = SimpleNamespace(
        status=SimpleNamespace(value=status),
        pipeline_id="test",
        completed_steps=0,
        total_steps=1,
        skipped_steps=0,
        failed_steps=0,
        duration_s=0,
        checkpoint_path="test.json",
        to_dict=lambda: {"status": status},
    )
    monkeypatch.setattr(PipelineExecutor, "execute", lambda *args, **kwargs: summary)
    result = CliRunner().invoke(
        cli, ["--cli", "pipeline", "run", "--case-dir", str(tmp_path), "--format", out_format]
    )
    assert result.exit_code == code
    assert status in result.output.lower()


@pytest.mark.parametrize("out_format", ["text", "json"])
@pytest.mark.parametrize(
    "status,code", [("failed", 1), ("partial", 1), ("cancelled", 1), ("completed", 0)]
)
def test_resume_exit_status_and_custom_recipe(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, out_format: str, status: str, code: int
) -> None:
    from lockknife.core.pipeline import PipelineExecutor, PlaybookDefinition, StepDefinition

    pb = PlaybookDefinition(
        "not-a-builtin",
        "Custom",
        "",
        "test",
        (StepDefinition("report", "Report", "report.generate", "test"),),
    )
    first = PipelineExecutor(case_dir=tmp_path).execute(pb)
    import dataclasses

    from lockknife.core.pipeline.models import PipelineStatus

    final = dataclasses.replace(first, status=PipelineStatus(status))
    seen = []

    def execute(self: object, playbook: PlaybookDefinition, **kwargs: object) -> object:
        seen.append(playbook.name)
        return final

    monkeypatch.setattr(PipelineExecutor, "execute", execute)
    result = CliRunner().invoke(
        cli,
        [
            "--cli",
            "pipeline",
            "resume",
            "--case-dir",
            str(tmp_path),
            "--pipeline-id",
            first.pipeline_id,
            "--format",
            out_format,
        ],
    )
    assert result.exit_code == code
    assert status in result.output.lower()
    assert seen == ["not-a-builtin"]


def test_cli_pipeline_list() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--cli", "pipeline", "list"])
    assert result.exit_code == 0
    assert "triage" in result.output
    assert "deep-forensics" in result.output
    assert "crypto-audit" in result.output


def test_cli_pipeline_list_json() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--cli", "pipeline", "list", "--format", "json"])
    assert result.exit_code == 0
    assert '"name": "triage"' in result.output


def test_cli_pipeline_plan() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--cli", "pipeline", "plan", "--playbook", "triage"])
    assert result.exit_code == 0
    assert "Rapid Security & Device Triage" in result.output
    assert "Execution Tier 0" in result.output


def test_cli_pipeline_run_dry_run_and_status(tmp_path: pathlib.Path) -> None:
    case_dir = tmp_path / "test-case-cli"
    runner = CliRunner()

    run_res = runner.invoke(
        cli,
        [
            "--cli",
            "pipeline",
            "run",
            "--playbook",
            "triage",
            "--case-dir",
            str(case_dir),
            "--dry-run",
        ],
    )
    assert run_res.exit_code == 0
    assert "Pipeline Summary: PENDING" in run_res.output

    status_res = runner.invoke(cli, ["--cli", "pipeline", "status", "--case-dir", str(case_dir)])
    assert status_res.exit_code == 0
    assert "No pipeline checkpoints found" in status_res.output


def test_cli_pipeline_resume(tmp_path: pathlib.Path) -> None:
    case_dir = tmp_path / "test-case-resume"
    runner = CliRunner()

    runner.invoke(
        cli,
        [
            "--cli",
            "pipeline",
            "run",
            "--playbook",
            "triage",
            "--case-dir",
            str(case_dir),
            "--dry-run",
        ],
    )
    resume_res = runner.invoke(cli, ["--cli", "pipeline", "resume", "--case-dir", str(case_dir)])
    assert resume_res.exit_code != 0
    assert "No pipeline checkpoints found" in resume_res.output
