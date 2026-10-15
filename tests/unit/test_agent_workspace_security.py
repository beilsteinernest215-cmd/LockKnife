from __future__ import annotations

import os
import pathlib

import pytest

from lockknife.core.agent import _workspace_fs
from lockknife.core.agent.memory import MemoryStore
from lockknife.core.agent.models import AgentGoal, ModelDecision, ToolInvocation, TurnRecord
from lockknife.core.agent.tools import AgentToolRegistry


@pytest.fixture
def workspace(tmp_path):
    case = tmp_path / "case"
    outside = tmp_path / "outside"
    case.mkdir()
    outside.mkdir()
    return case, outside


def test_hardlink_reads_are_denied_and_writes_replace_only_case_link(workspace):
    case, outside = workspace
    original = outside / "secret"
    original.write_text("outside original")
    os.link(original, case / "linked")
    tools = AgentToolRegistry(case_dir=case)
    read = tools.execute(ToolInvocation("read", "case_read_file", {"relative_path": "linked"}))
    assert not read.success
    assert "single-link" in read.error
    write = tools.execute(
        ToolInvocation(
            "write", "case_write_file", {"relative_path": "linked", "content": "case replacement"}
        )
    )
    assert write.success
    assert original.read_text() == "outside original"
    assert (case / "linked").read_text() == "case replacement"
    assert (case / "linked").stat().st_ino != original.stat().st_ino


@pytest.mark.skipif(os.name != "posix", reason="Controlled descriptor-open swap uses POSIX dir_fd")
@pytest.mark.parametrize("operation", ["case_read_file", "case_write_file"])
@pytest.mark.parametrize("swap_after_open", [False, True])
def test_directory_swap_cannot_redirect_file_access(
    monkeypatch, workspace, operation, swap_after_open
):
    case, outside = workspace
    folder = case / "data"
    folder.mkdir()
    (folder / "file").write_text("inside")
    (outside / "file").write_text("outside secret")
    tools = AgentToolRegistry(case_dir=case)
    actual_open = os.open
    swapped = False

    def intercepted(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "data" and dir_fd is not None and flags & os.O_DIRECTORY and not swapped:
            swapped = True
            fd = actual_open(path, flags, mode, dir_fd=dir_fd) if swap_after_open else None
            folder.rename(case / "saved")
            folder.symlink_to(outside, target_is_directory=True)
            if fd is not None:
                return fd
        return actual_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(_workspace_fs.os, "open", intercepted)
    result = tools.execute(
        ToolInvocation("call", operation, {"relative_path": "data/file", "content": "new inside"})
    )
    assert swapped
    assert (outside / "file").read_text() == "outside secret"
    if operation == "case_read_file" and result.success:
        assert result.output == "inside"
    if operation == "case_write_file" and result.success:
        assert (case / "saved" / "file").read_text() == "new inside"


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
@pytest.mark.parametrize("operation", ["set_fact", "read", "turn", "spill"])
def test_memory_rejects_replaced_agent_ancestor(workspace, replacement, operation):
    case, outside = workspace
    memory = MemoryStore(case)
    memory.set_fact("original", "preserved")
    memory.agent_dir.rename(case / "saved-agent")
    if replacement == "symlink":
        memory.agent_dir.symlink_to(outside, target_is_directory=True)
    else:
        memory.agent_dir.mkdir()
    with pytest.raises(OSError):
        if operation == "set_fact":
            memory.set_fact("attack", "blocked")
        elif operation == "read":
            memory.get_facts()
        elif operation == "turn":
            memory.persist_turn(TurnRecord(1, ModelDecision.respond("answer")))
        else:
            memory.truncate_observation_payload("artifact", "x" * 5000)
    assert list(outside.iterdir()) == []
    assert not (memory.agent_dir / "facts.sqlite3").exists()
    assert (case / "saved-agent" / "facts.json").read_text().find("preserved") >= 0


def test_memory_rejects_hardlinked_facts_and_lock_files(workspace):
    case, outside = workspace
    memory = MemoryStore(case)
    secret = outside / "secret.json"
    secret.write_text('{"private":"secret"}')
    os.link(secret, memory.facts_path)
    with pytest.raises(PermissionError, match="single-link"):
        memory.get_facts()
    memory.facts_path.unlink()
    os.link(secret, memory.agent_dir / "facts.lock")
    with pytest.raises(PermissionError, match="single-link"):
        memory.set_fact("blocked", "blocked")
    assert secret.read_text() == '{"private":"secret"}'


def test_memory_swap_after_fact_lock_never_writes_outside(monkeypatch, workspace):
    case, outside = workspace
    memory = MemoryStore(case)
    memory.set_fact("original", "retained")
    write_facts = memory._write_facts

    def replace_before_commit():
        memory.agent_dir.rename(case / "saved-agent")
        memory.agent_dir.symlink_to(outside, target_is_directory=True)
        write_facts()

    monkeypatch.setattr(memory, "_write_facts", replace_before_commit)
    with pytest.raises(OSError):
        memory.set_fact("new", "blocked")
    assert list(outside.iterdir()) == []
    assert (case / "saved-agent" / "facts.json").read_text().find("retained") >= 0


@pytest.mark.skipif(os.name != "posix", reason="Controlled descriptor-open swap uses POSIX dir_fd")
def test_leaf_swap_before_open_is_not_followed(monkeypatch, workspace):
    case, outside = workspace
    target = case / "read.txt"
    target.write_text("inside")
    secret = outside / "secret"
    secret.write_text("outside secret")
    tools = AgentToolRegistry(case_dir=case)
    actual_open = os.open

    def replaced_leaf(path, flags, mode=0o777, *, dir_fd=None):
        if path == "read.txt" and dir_fd is not None:
            target.unlink()
            target.symlink_to(secret)
        return actual_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(_workspace_fs.os, "open", replaced_leaf)
    result = tools.execute(ToolInvocation("read", "case_read_file", {"relative_path": "read.txt"}))
    assert not result.success
    assert result.output is None


@pytest.mark.parametrize(
    "path", [r"\outside\file", "C:relative", "D:/outside", "../outside", "note:stream", "note. "]
)
def test_windows_path_forms_cannot_reset_the_anchor(monkeypatch, path):
    monkeypatch.setattr(_workspace_fs, "_POSIX", False)
    files = object.__new__(_workspace_fs.WorkspaceFiles)
    files.root = pathlib.PureWindowsPath("C:/case")
    with pytest.raises(PermissionError):
        files._parts(path)
    assert files._parts("C:/case/nested/../note") == ("note",)


def test_ordinary_atomic_memory_and_nested_case_paths_remain_usable(workspace):
    case, _ = workspace
    memory = MemoryStore(case)
    memory.set_fact("fact", "value")
    memory.persist_turn(TurnRecord(1, ModelDecision.respond("answer")))
    from lockknife.core.agent.models import AgentRunResult, GoalStatus

    memory.persist_result(AgentRunResult(AgentGoal("normal"), GoalStatus.COMPLETED, "done"))
    _, spill = memory.truncate_observation_payload("safe", "x" * 5000)
    assert pathlib.Path(spill).read_text() == "x" * 5000
    assert memory.get_facts() == {"fact": "value"}
    assert memory.load_episodic_turns()[0]["decision"]["text"] == "answer"
    tools = AgentToolRegistry(case_dir=case)
    assert tools.execute(
        ToolInvocation(
            "w", "case_write_file", {"relative_path": "notes/../nested/note", "content": "ok"}
        )
    ).success
    assert (
        tools.execute(
            ToolInvocation("r", "case_read_file", {"relative_path": str(case / "nested" / "note")})
        ).output
        == "ok"
    )
