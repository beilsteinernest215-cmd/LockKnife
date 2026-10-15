from __future__ import annotations

import pathlib
import subprocess
from unittest.mock import Mock

import pytest

from lockknife.modules.exploitation.wifi import mitm


@pytest.fixture
def config():
    return mitm.MITMConfig(mitm.MITMMethod.ARP_SPOOF, "192.0.2.2", "192.0.2.1")


@pytest.fixture
def process(monkeypatch):
    proc = Mock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    monkeypatch.setattr(mitm.time, "sleep", lambda _: None)
    monkeypatch.setattr(mitm.subprocess, "Popen", Mock(return_value=proc))
    monkeypatch.setattr(mitm.subprocess, "run", Mock(side_effect=AssertionError("global cleanup")))
    return proc


def test_startup_is_not_verified_interception(config, process):
    attack = mitm.MITMAttack()
    result = attack.start(config)
    assert result.success
    assert result.config is config
    assert result.state is mitm.MITMState.RUNNING
    assert result.metadata["process_started"] is True
    assert result.metadata["interception_verified"] is False
    assert result.bytes_intercepted == result.connections_intercepted == 0
    kwargs = mitm.subprocess.Popen.call_args.kwargs
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] != subprocess.PIPE
    assert attack.stop()
    process.terminate.assert_called_once()
    process.wait.assert_called_once()


@pytest.mark.parametrize("method", [mitm.MITMMethod.DHCP_SPOOF, mitm.MITMMethod.HTTP_INJECT])
@pytest.mark.parametrize("dry_run", [False, True])
def test_placeholders_are_unsupported(config, process, method, dry_run):
    config.method = method
    result = mitm.MITMAttack().start(config, dry_run=dry_run)
    assert not result.success
    assert result.state is mitm.MITMState.FAILED
    assert result.metadata["unsupported"]
    assert "unsupported" in result.error
    mitm.subprocess.Popen.assert_not_called()


def test_dry_run_has_no_fake_counters_or_running_state(config, process):
    result = mitm.MITMAttack().start(config, dry_run=True)
    assert result.success
    assert result.state is mitm.MITMState.NOT_STARTED
    assert not result.metadata["process_started"]
    assert result.bytes_intercepted == result.connections_intercepted == 0
    mitm.subprocess.Popen.assert_not_called()


def test_startup_exit_reports_stderr_and_code(config, process, monkeypatch):
    process.poll.return_value = 2

    def launch(*args, **kwargs):
        kwargs["stderr"].write(b"permission denied")
        return process

    monkeypatch.setattr(mitm.subprocess, "Popen", launch)
    attack = mitm.MITMAttack()
    result = attack.start(config)
    assert not result.success
    assert result.state is mitm.MITMState.FAILED
    assert "code 2" in result.error
    assert "permission denied" in result.error
    assert attack._mitm_process is None
    process.wait.assert_called_once()


def test_missing_sslstrip_is_not_replaced_with_generic_proxy(config, process):
    config.method = mitm.MITMMethod.SSL_STRIP
    mitm.subprocess.Popen.side_effect = FileNotFoundError("sslstrip")
    result = mitm.MITMAttack().start(config)
    assert not result.success
    assert "Missing dependency" in result.error
    assert mitm.subprocess.Popen.call_count == 1


def test_arp_fallback_retains_config_and_truth(config, process):
    mitm.subprocess.Popen.side_effect = [FileNotFoundError("arpspoof"), process]
    result = mitm.MITMAttack(ettercap_path="/test/ettercap").start(config)
    assert result.success
    assert mitm.subprocess.Popen.call_args.args[0][0] == "/test/ettercap"
    assert not result.metadata["interception_verified"]


def test_stop_kills_then_reaps_owned_process(config, process):
    attack = mitm.MITMAttack()
    attack.start(config)
    process.wait.side_effect = [subprocess.TimeoutExpired("arpspoof", 5), 0]
    assert attack.stop()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2
    assert attack._mitm_process is None


def test_failed_stop_retains_ownership(config, process):
    attack = mitm.MITMAttack()
    attack.start(config)
    process.wait.side_effect = OSError("unreaped")
    assert not attack.stop()
    assert attack._mitm_process is process


def test_duplicate_start_does_not_lose_process(config, process):
    attack = mitm.MITMAttack()
    assert attack.start(config).success
    assert not attack.start(config).success
    assert mitm.subprocess.Popen.call_count == 1
    assert attack._mitm_process is process


@pytest.mark.parametrize("exit_code", [None, 7])
def test_duration_result_reflects_stop_or_early_exit(config, process, exit_code):
    process.poll.side_effect = [None, exit_code, exit_code]
    result = mitm.MITMAttack().start(config, duration_s=1)
    assert result.success is (exit_code is None)
    assert result.state is (mitm.MITMState.STOPPED if exit_code is None else mitm.MITMState.FAILED)


@pytest.mark.parametrize("startup_failure", [False, True])
def test_dns_hosts_file_is_removed(config, process, startup_failure):
    config.method = mitm.MITMMethod.DNS_SPOOF
    config.dns_mappings = {"example.test": "192.0.2.3"}
    if startup_failure:
        mitm.subprocess.Popen.side_effect = FileNotFoundError("dnsspoof")
    attack = mitm.MITMAttack()
    result = attack.start(config)
    hosts = pathlib.Path(mitm.subprocess.Popen.call_args.args[0][-1])
    if startup_failure:
        assert not result.success
        assert "Missing dependency" in result.error
    else:
        assert hosts.read_text() == "192.0.2.3 example.test\n"
        assert attack.stop()
    assert not hosts.exists()


def test_combined_options_are_not_silently_ignored(config, process):
    config.enable_dns_spoof = True
    result = mitm.MITMAttack().start(config)
    assert not result.success
    assert result.metadata["unsupported"]
    mitm.subprocess.Popen.assert_not_called()
