from __future__ import annotations

import json
import shlex
import sys
from collections import OrderedDict

import pytest

from lockknife.core.agent.models import AgentGoal, GoalStatus, ModelDecision, ToolInvocation
from lockknife.core.agent.provider import DeterministicMockProvider
from lockknife.core.agent.runtime import AutonomousRuntime
from lockknife_headless_cli import _tui_callback_agent as chat
from lockknife_headless_cli.tui_callback import build_tui_callback


@pytest.fixture(autouse=True)
def isolated_chat_cache(monkeypatch):
    chat._close_chat_sessions()
    monkeypatch.setattr(chat, "_chat_sessions", OrderedDict())
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("No network allowed"))
    yield
    chat._close_chat_sessions()


def test_tui_preserves_two_user_turns_and_finishes_tools_in_one_input(monkeypatch, tmp_path):
    provider = DeterministicMockProvider(
        [
            ModelDecision.call_tools(
                [ToolInvocation("fact", "record_fact", {"key": "finding", "value": "alpha"})]
            ),
            ModelDecision.respond("First answer after recording the finding."),
            ModelDecision.respond("Second answer remembers alpha."),
        ]
    )
    created = []

    def make_provider(config):
        created.append(config)
        return provider

    monkeypatch.setattr("lockknife.core.agent.OpenAICompatibleProvider", make_provider)
    callback = build_tui_callback(None)
    first = callback(
        "agent.chat", {"prompt": "Remember alpha", "case_dir": str(tmp_path), "serial": "target"}
    )
    assert first["ok"]
    assert "First answer after recording" in json.loads(first["data_json"])["response"]
    assert len(provider.call_history) == 2
    assert provider.call_history[1]["messages"][-1]["role"] == "tool"
    second = callback(
        "agent.chat",
        {"prompt": "What did we find?", "case_dir": str(tmp_path / "."), "target_device": "target"},
    )
    assert second["ok"]
    assert len(created) == 1
    messages = provider.call_history[2]["messages"]
    assert {"role": "user", "content": "Remember alpha"} in messages
    assert {"role": "assistant", "content": "First answer after recording the finding."} in messages
    assert messages[-1] == {"role": "user", "content": "What did we find?"}
    runtime = next(iter(chat._chat_sessions.values())).runtime
    assert runtime.goal.budget_iterations == 25
    assert runtime.memory.get_facts()["finding"] == "alpha"
    turns = runtime.memory.load_episodic_turns()
    assert len(turns) == 3
    assert [turn["user_input"] for turn in turns] == ["Remember alpha", None, "What did we find?"]


def test_chat_continues_reasoning_and_persists_provider_error(tmp_path):
    provider = DeterministicMockProvider(
        [
            ModelDecision.continue_run("Still reasoning"),
            ModelDecision.respond("Final answer"),
            ModelDecision.failure("Provider unavailable"),
        ]
    )
    runtime = AutonomousRuntime(
        goal=AgentGoal("chat", budget_iterations=5), case_dir=tmp_path, provider=provider
    )
    assert runtime.chat_step("First question") == "Final answer"
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        runtime.chat_step("Second question")
    assert runtime.goal.status == GoalStatus.FAILED
    assert len(runtime.memory.load_episodic_turns()) == 3


def test_tui_budget_exhaustion_is_not_success_or_a_fresh_session(monkeypatch, tmp_path):
    provider = DeterministicMockProvider(
        [
            ModelDecision.call_tools(
                [ToolInvocation("fact", "record_fact", {"key": "recorded", "value": "yes"})]
            ),
            ModelDecision.respond("This answer must not be requested"),
        ]
    )
    created = []

    def make_provider(config):
        created.append(config)
        return provider

    monkeypatch.setattr("lockknife.core.agent.OpenAICompatibleProvider", make_provider)
    callback = build_tui_callback(None)
    params = {"prompt": "Do work", "case_dir": str(tmp_path), "budget": 1}
    first = callback("agent.chat", params)
    assert not first["ok"]
    assert "budget exhausted" in json.dumps(first)
    second = callback("agent.chat", params)
    assert not second["ok"]
    assert len(created) == len(provider.call_history) == 1
    runtime = next(iter(chat._chat_sessions.values())).runtime
    assert runtime.goal.status == GoalStatus.BUDGET_EXHAUSTED
    assert runtime.memory.get_facts()["recorded"] == "yes"
    assert len(runtime.memory.load_episodic_turns()) == 3


def test_cache_bounds_eviction_idle_cleanup_and_configuration_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "_CHAT_LIMIT", 2)
    callback = build_tui_callback(None)
    common = {"case_dir": str(tmp_path), "mock": True, "prompt": "hello"}
    assert callback("agent.chat", {**common, "model": "model-a"})["ok"]
    first = next(iter(chat._chat_sessions.values())).runtime
    session = first.tools.exec_sessions.start_session(
        shlex.join([sys.executable, "-c", "import time;time.sleep(30)"])
    )
    assert session["ok"]
    assert callback("agent.chat", {**common, "model": "model-b"})["ok"]
    assert callback("agent.chat", {**common, "model": "model-b", "serial": "different"})["ok"]
    assert len(chat._chat_sessions) == 2
    assert not first.tools.exec_sessions.list_sessions()
    stale = next(iter(chat._chat_sessions.values()))
    stale.last_used -= chat._CHAT_IDLE_SECONDS + 1
    assert callback("agent.chat", {**common, "model": "model-c"})["ok"]
    assert stale not in chat._chat_sessions.values()


def test_mock_chat_responds_to_each_current_prompt(tmp_path):
    callback = build_tui_callback(None)
    common = {"case_dir": str(tmp_path), "mock": True}
    for prompt in ["first prompt", "second prompt"]:
        result = callback("agent.chat", {**common, "prompt": prompt})
        assert result["ok"]
        assert prompt in json.loads(result["data_json"])["response"]
    assert len(chat._chat_sessions) == 1


def test_cache_isolated_by_app_identity(tmp_path):
    owners = [object(), object()]
    made = []

    def factory():
        runtime = AutonomousRuntime(
            goal=AgentGoal("chat"), case_dir=tmp_path, provider=DeterministicMockProvider()
        )
        made.append(runtime)
        return runtime

    for owner in owners:
        with chat._chat_runtime((id(owner), str(tmp_path)), owner, factory) as runtime:
            assert runtime.chat_step("hello")
    assert len(made) == 2
