from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest
from click.testing import CliRunner

from lockknife.core.agent.exec_session import ExecSessionManager
from lockknife.core.agent.failover import FailoverProvider
from lockknife.core.agent.memory import MemoryStore
from lockknife.core.agent.models import (
    AgentGoal,
    DecisionKind,
    GoalStatus,
    ModelDecision,
    ToolInvocation,
)
from lockknife.core.agent.provider import DeterministicMockProvider, OpenAICompatibleProvider
from lockknife.core.agent.runtime import AutonomousRuntime
from lockknife.core.agent.subagent import ExecutionBudget
from lockknife.core.agent.tools import AgentToolRegistry


def response(message, finish_reason="stop"):
    return io.BytesIO(
        json.dumps({"choices": [{"message": message, "finish_reason": finish_reason}]}).encode()
    )


@pytest.mark.parametrize("tool", ["case_read_file", "case_write_file"])
def test_case_containment_equivalent_and_symlink_paths(tmp_path, tool):
    case = tmp_path / "case"
    outside = tmp_path / "case-sibling"
    case.mkdir()
    outside.mkdir()
    (outside / "secret").write_text("outside")
    (case / "link").symlink_to(outside, target_is_directory=True)
    registry = AgentToolRegistry(case_dir=case)
    for path in ["../case-sibling/secret", "link/secret", str(outside / "secret")]:
        result = registry.execute(
            ToolInvocation("call", tool, {"relative_path": path, "content": "changed"})
        )
        assert not result.success
    assert (outside / "secret").read_text() == "outside"
    written = registry.execute(
        ToolInvocation(
            "write", "case_write_file", {"relative_path": "./notes/../note", "content": "inside"}
        )
    )
    assert written.success
    read = registry.execute(
        ToolInvocation("read", "case_read_file", {"relative_path": str(case / "note")})
    )
    assert read.output == "inside"


def test_real_provider_alias_roundtrip_overflow_and_runtime(monkeypatch, tmp_path):
    requests = []

    def external_io(request, **kwargs):
        payload = json.loads(request.data)
        requests.append(payload)
        assert len(payload["tools"]) <= 128
        assert all(
            re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool["function"]["name"])
            for tool in payload["tools"]
        )
        if len(requests) == 1:
            tool = next(t for t in payload["tools"] if t["function"]["name"] == "case_write_file")
            return response(
                {
                    "tool_calls": [
                        {
                            "id": "write",
                            "function": {
                                "name": tool["function"]["name"],
                                "arguments": json.dumps(
                                    {"relative_path": "proof.txt", "content": "verified"}
                                ),
                            },
                        }
                    ]
                }
            )
        assert payload["messages"][-1]["role"] == "tool"
        return response({"content": "done"})

    monkeypatch.setattr("urllib.request.urlopen", external_io)
    runtime = AutonomousRuntime(
        goal=AgentGoal("Write evidence"),
        case_dir=tmp_path / "case",
        provider=OpenAICompatibleProvider(),
    )
    result = runtime.run()
    assert result.status == GoalStatus.COMPLETED
    assert (runtime.case_dir / "proof.txt").read_text() == "verified"
    assert len(runtime.memory.load_episodic_turns()) == 2

    specs = [
        {"type": "function", "function": {"name": f"action.{i}", "parameters": {"type": "object"}}}
        for i in range(140)
    ]

    def overflow_io(request, **kwargs):
        payload = json.loads(request.data)
        dispatcher = payload["tools"][-1]["function"]
        assert "action.139" in dispatcher["parameters"]["properties"]["action"]["enum"]
        return response(
            {
                "tool_calls": [
                    {
                        "id": "overflow",
                        "function": {
                            "name": dispatcher["name"],
                            "arguments": json.dumps(
                                {"action": "action.139", "arguments": {"x": 1}}
                            ),
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", overflow_io)
    decision = OpenAICompatibleProvider().complete([], tools=specs)
    assert decision.tool_calls[0].tool_id == "action.139"
    assert decision.tool_calls[0].arguments == {"x": 1}
    assert specs[0]["function"]["name"] == "action.0"


def test_continue_delegation_and_isolated_durable_runs(tmp_path):
    provider = DeterministicMockProvider(
        [
            ModelDecision.continue_run("still working"),
            ModelDecision.call_tools(
                [ToolInvocation("child", "delegate_subagent", {"objective": "child"})]
            ),
            ModelDecision.finish("child done"),
            ModelDecision.finish("parent done"),
        ]
    )
    runtime = AutonomousRuntime(goal=AgentGoal("parent"), case_dir=tmp_path, provider=provider)
    result = runtime.run()
    assert result.status == GoalStatus.COMPLETED
    assert len(result.turns) == 3
    assert result.turns[1].observations[0].success
    child = next(iter(runtime.subagent_manager.active_subagents.values()))
    assert child.status == GoalStatus.COMPLETED
    turns = runtime.memory.load_episodic_turns()
    assert len(turns) == 4
    assert len({t["run_id"] for t in turns}) == 2
    assert len({t["goal_id"] for t in turns}) == 2
    assert len(list(runtime.memory.memory_dir.glob("result_*.json"))) == 2


def test_concurrent_fact_writes_preserve_all_stores(tmp_path):
    stores = [MemoryStore(tmp_path) for _ in range(4)]

    def write(index):
        stores[index % 4].set_fact(str(index), index)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(80)))
    assert MemoryStore(tmp_path).get_facts() == {str(i): i for i in range(80)}
    stores[0].set_fact("0", "revised")
    entries = [json.loads(path.read_text()) for path in stores[0].memory_dir.glob("fact_*.json")]
    assert {str(entry["value"]) for entry in entries if entry["key"] == "0"} == {"0", "revised"}


def test_provider_failure_retry_failover_and_cli_exit(monkeypatch, tmp_path):
    attempts = []

    def failed_io(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.HTTPError(
            "http://provider", 503, "Unavailable", {}, io.BytesIO(b"secret response")
        )

    monkeypatch.setattr("urllib.request.urlopen", failed_io)
    primary = OpenAICompatibleProvider()
    runtime = AutonomousRuntime(
        goal=AgentGoal("test failure"), case_dir=tmp_path / "failed", provider=primary
    )
    result = runtime.run()
    assert result.status == GoalStatus.FAILED
    assert len(attempts) == 3
    assert "secret response" not in result.error
    fallback = FailoverProvider(
        [primary, DeterministicMockProvider([ModelDecision.respond("recovered")])]
    )
    assert fallback.complete([]).text == "recovered"
    from lockknife_headless_cli.agent import agent_group

    for output_format in ["json", "text", "markdown"]:
        cli = CliRunner().invoke(
            agent_group,
            [
                "goal",
                "test",
                "--case-dir",
                str(tmp_path / output_format),
                "--format",
                output_format,
            ],
        )
        assert cli.exit_code == 1, cli.output


@pytest.mark.parametrize("payload", [{}, [], {"choices": [3]}, {"choices": [{"message": "bad"}]}])
def test_malformed_provider_is_failure(monkeypatch, payload):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode())
    )
    assert OpenAICompatibleProvider().complete([]).kind == DecisionKind.ERROR


def test_exec_quoted_arguments_and_partial_output():
    manager = ExecSessionManager()
    command = shlex.join(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys,time;sys.stdout.write('quoted value');sys.stdout.flush();time.sleep(30)",
        ]
    )
    try:
        assert manager.start_session(command, "quoted")["ok"]
        output = ""
        deadline = time.monotonic() + 5
        while "quoted value" not in output and time.monotonic() < deadline:
            output += manager.poll_session("quoted", wait_s=0.05)["output"]
        assert output == "quoted value"
        assert not manager.write_session("quoted", "x" * 65537)["ok"]
        process = manager._sessions["quoted"].proc
    finally:
        manager.close_all()
    assert process.poll() is not None


def test_provider_alias_history_and_nonretryable_failure(monkeypatch):
    specifications = [{"type": "function", "function": {"name": "core.health"}}]
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "one",
                    "function": {
                        "name": "core.health",
                        "arguments": "{}",
                    },
                }
            ],
        }
    ]

    def transport(request, **kwargs):
        payload = json.loads(request.data)
        alias = payload["tools"][0]["function"]["name"]
        assert payload["messages"][0]["tool_calls"][0]["function"]["name"] == alias
        return response(
            {"tool_calls": [{"id": "two", "function": {"name": alias, "arguments": "{}"}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", transport)
    assert (
        OpenAICompatibleProvider().complete(history, specifications).tool_calls[0].tool_id
        == "core.health"
    )
    assert history[0]["tool_calls"][0]["function"]["name"] == "core.health"
    calls = []

    def unauthorized(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("http://provider", 401, "Unauthorized", {}, io.BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", unauthorized)
    assert OpenAICompatibleProvider().complete([]).kind == DecisionKind.ERROR
    assert len(calls) == 1
    result = FailoverProvider(
        [DeterministicMockProvider([ModelDecision.finish("Report: quota exceeded yesterday")])]
    ).complete([])
    assert result.kind == DecisionKind.FINISH


def test_tui_real_provider_default_and_daemon_lifecycle(monkeypatch, tmp_path):
    from lockknife.core.adb import AdbClient
    from lockknife_headless_cli import _tui_callback_agent
    from lockknife_headless_cli.tui_callback import build_tui_callback

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: response({"content": "real transport used"})
    )
    monkeypatch.setattr(AdbClient, "run", lambda *a, **k: "List of devices attached\n")
    monkeypatch.setattr(_tui_callback_agent, "_daemon", None)
    callback = build_tui_callback(None)
    result = callback("agent.chat", {"prompt": "hello", "case_dir": str(tmp_path)})
    assert result["ok"]
    assert "real transport used" in json.dumps(result)
    assert (
        json.loads(callback("agent.daemon", {"action": "status"})["data_json"])["status"]
        == "stopped"
    )
    try:
        assert (
            json.loads(callback("agent.daemon", {"action": "start", "interval": 60})["data_json"])[
                "status"
            ]
            == "active"
        )
        first = _tui_callback_agent._daemon
        callback("agent.daemon", {"action": "start"})
        assert _tui_callback_agent._daemon is first
        assert (
            json.loads(callback("agent.daemon", {"action": "status"})["data_json"])["status"]
            == "active"
        )
    finally:
        result = callback("agent.daemon", {"action": "stop"})
    assert json.loads(result["data_json"])["status"] == "stopped"


@pytest.mark.parametrize("tool_id", ["delegate_subagent", "agent.goal", "agent.chat"])
def test_nested_delegation_has_one_atomic_budget_and_depth_bound(tmp_path, tool_id):
    provider = DeterministicMockProvider(
        decision_hook=lambda _: ModelDecision.call_tools(
            [
                ToolInvocation(
                    "delegate",
                    tool_id,
                    {"objective": "recurse", "budget_iterations": 30},
                )
            ]
        )
    )
    budget = ExecutionBudget(7, max_depth=2)
    runtime = AutonomousRuntime(
        goal=AgentGoal("root"), case_dir=tmp_path, provider=provider, execution_budget=budget
    )
    result = runtime.run()
    assert result.status == GoalStatus.BUDGET_EXHAUSTED
    assert len(provider.call_history) == 7
    assert budget.remaining == 0
    history = runtime.memory.load_episodic_turns()
    assert any("depth limit" in str(turn) for turn in history)
    atomic = ExecutionBudget(10)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: atomic.consume(), range(100))) == 10


def test_overflow_collision_and_history_routing(monkeypatch):
    collision = "lk_" + hashlib.sha256(b"core.health").hexdigest()[:60]
    names = ["core.health", collision, "lk_overflow"] + [f"action.{i}" for i in range(140)]
    specs = [{"type": "function", "function": {"name": name}} for name in names]
    calls = []

    def transport(request, **kwargs):
        payload = json.loads(request.data)
        definitions = payload["tools"]
        aliases = [item["function"]["name"] for item in definitions]
        assert len(aliases) == len(set(aliases)) == 128
        assert aliases[0] != collision
        dispatcher = definitions[-1]["function"]
        assert dispatcher["name"] != "lk_overflow"
        if calls:
            past = payload["messages"][0]["tool_calls"][0]["function"]
            assert past["name"] == dispatcher["name"]
            assert json.loads(past["arguments"]) == {"action": "action.139", "arguments": {"x": 1}}
        calls.append(1)
        return response(
            {
                "tool_calls": [
                    {
                        "id": "one",
                        "function": {
                            "name": dispatcher["name"],
                            "arguments": json.dumps(
                                {"action": "action.139", "arguments": {"x": 1}}
                            ),
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", transport)
    provider = OpenAICompatibleProvider()
    first = provider.complete([], specs)
    assert first.tool_calls[0].tool_id == "action.139"
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "one", "function": {"name": "action.139", "arguments": '{"x":1}'}}
            ],
        }
    ]
    assert provider.complete(history, specs).tool_calls[0].tool_id == "action.139"


def test_windows_pipe_fallback_with_real_process(monkeypatch):
    import subprocess

    from lockknife.core.agent import exec_session

    monkeypatch.setattr(exec_session, "_POSIX", False)

    def unsupported(*args, **kwargs):
        raise AssertionError("Unix-only pipe operation called")

    monkeypatch.setattr(exec_session.os, "set_blocking", unsupported)
    monkeypatch.setattr(exec_session.select, "select", unsupported)
    monkeypatch.setattr(exec_session.os, "killpg", unsupported)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys,time;print(sys.stdin.readline().strip(),flush=True);time.sleep(30)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    session = exec_session.ExecSession("windows", "echo", proc)
    try:
        assert session.write_input("pipe fallback")
        deadline = time.monotonic() + 5
        output = ""
        while not output and time.monotonic() < deadline:
            output += session.poll_output(0.01)["output"]
        assert "pipe fallback" in output
    finally:
        session.close()
    assert proc.poll() is not None


def test_delegation_inherits_parent_target(tmp_path):
    provider = DeterministicMockProvider([ModelDecision.finish("child done")])
    runtime = AutonomousRuntime(
        goal=AgentGoal("parent"),
        case_dir=tmp_path,
        target_device="parent-serial",
        provider=provider,
    )
    result = runtime.subagent_manager.spawn_and_run("child")
    assert result.status == GoalStatus.COMPLETED
    assert result.goal.target_device == "parent-serial"
    assert "Target Device: parent-serial" in provider.call_history[0]["messages"][0]["content"]


def test_exec_logs_exclude_commands(caplog):
    import logging

    manager = ExecSessionManager()
    secret = "credential-not-for-logs"
    with caplog.at_level(logging.INFO, logger="lockknife.agent.exec_session"):
        try:
            result = manager.start_session(
                shlex.join([sys.executable, "-c", f"print('{secret}')"]), "safe-log"
            )
            assert result["ok"]
        finally:
            manager.close_all()
        manager.start_session(f"/nonexistent/{secret}", "failed-log")
    assert secret not in caplog.text
    assert "safe-log" in caplog.text
    assert "failed-log" in caplog.text
