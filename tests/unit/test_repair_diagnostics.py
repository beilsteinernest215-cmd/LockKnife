from __future__ import annotations

import sys

import click
import pytest
from click.testing import CliRunner

from lockknife.core.cli_instrumentation import LockKnifeCommand
from lockknife.core.config import LockKnifeConfig
from lockknife.core.error_report import capture
from lockknife.core.health import _check_module, install_missing_dependencies
from lockknife.core.log_redaction import credential_context, redact
from lockknife.core.logging import configure_logging, get_logger


@pytest.mark.parametrize("log_format", ["json", "console"])
@pytest.mark.parametrize("fail", [False, True])
def test_cli_credentials_never_reach_logs_or_error_messages(log_format, fail):
    secret = "DUMMY-" + "PRIVATE-CREDENTIAL"
    configure_logging(LockKnifeConfig(log_level="INFO", log_format=log_format))

    @click.command(cls=LockKnifeCommand)
    @click.option("--api-key")
    def command(api_key):
        get_logger().info("nested", payload={"accessToken": api_key})
        if fail:
            raise click.ClickException(f"Request failed with {api_key}")

    result = CliRunner().invoke(command, ["--api-key", secret])
    assert result.exit_code == int(fail)
    assert secret not in result.output
    assert "[REDACTED]" in result.output


def test_error_report_redacts_traceback_and_restores_context():
    secret = "DUMMY-TRACE-CREDENTIAL"
    with credential_context({"password": secret}):
        try:
            raise RuntimeError(secret)
        except RuntimeError:
            report = capture("dummy", extra={"apiKey": secret})
        assert secret not in report.traceback
        assert secret not in report.exc_message
        assert report.extra == {"apiKey": "[REDACTED]"}
    assert redact(secret) == secret


def test_module_diagnostics_do_not_mutate_search_path(monkeypatch, tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    (root / "dummy_diagnostic_module.py").write_text("raise RuntimeError('must not load')")
    monkeypatch.setattr("lockknife.core.health.get_fallback_site_packages", lambda: [str(root)])
    before = list(sys.path)
    result = _check_module("dummy_diagnostic_module")
    assert not result["ok"]
    assert "Found on disk" in result["error"]
    assert before == sys.path
    assert "dummy_diagnostic_module" not in sys.modules


@pytest.mark.parametrize("kwargs", [{}, {"all_extras": True}, {"extras": ["apk"]}])
def test_install_preview_never_runs_diagnostics_or_subprocess(monkeypatch, kwargs):
    def forbidden(*args, **kw):
        raise AssertionError("Dry-run attempted an external operation")

    monkeypatch.setattr("lockknife.core.health.doctor_status", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    result = install_missing_dependencies(dry_run=True, **kwargs)
    assert result["ok"]
    assert result["dry_run"]
    assert not result["installed"]
