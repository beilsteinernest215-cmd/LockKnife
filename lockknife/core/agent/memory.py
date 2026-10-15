from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import pathlib
import re
import time
import uuid
from typing import Any

from lockknife.core.agent._workspace_fs import WorkspaceFiles
from lockknife.core.agent.models import AgentGoal, AgentRunResult, TurnRecord

logger = logging.getLogger("lockknife.agent.memory")

_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_CONTEXT_TOKENS = 12000
_MAX_OBSERVATION_CHARS = 4000


@dataclasses.dataclass
class WorkingMemory:
    """Active in-memory working state for the current run."""

    goal: AgentGoal
    scratchpad: list[str] = dataclasses.field(default_factory=list)
    recent_turns: list[TurnRecord] = dataclasses.field(default_factory=list)

    def add_scratchpad_note(self, note: str) -> None:
        self.scratchpad.append(note.strip())


class MemoryStore:
    """Three-tier memory manager: Working, Episodic (CaseStore), and Semantic Facts."""

    def __init__(
        self,
        case_dir: pathlib.Path,
        max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
        goal_id: str | None = None,
    ) -> None:
        self.case_dir = pathlib.Path(case_dir).resolve()
        self.max_context_tokens = max_context_tokens
        self.goal_id = goal_id
        self.run_id = uuid.uuid4().hex
        self.agent_dir = self.case_dir / ".agent"
        self.memory_dir = self.agent_dir / "memory"
        self.artifacts_dir = self.agent_dir / "artifacts"
        self.facts_path = self.agent_dir / "facts.json"

        self.case_dir.mkdir(parents=True, exist_ok=True)
        self._files = WorkspaceFiles(self.case_dir)
        for directory in (".agent", ".agent/memory", ".agent/artifacts"):
            self._files.ensure_directory(directory, pin=True)

        self._facts: dict[str, Any] = self._load_facts()

    @contextlib.contextmanager
    def _facts_lock(self):
        with self._files.lock_file(".agent/facts.lock"):
            yield

    def _write_facts(self) -> None:
        self._files.write_text(self.facts_path, json.dumps(self._facts, indent=2))

    def _load_facts(self) -> dict[str, Any]:
        try:
            return json.loads(self._files.read_text(self.facts_path))
        except FileNotFoundError:
            return {}

    def save_facts(self) -> None:
        with self._facts_lock():
            self._facts = {**self._load_facts(), **self._facts}
            self._write_facts()

    def set_fact(self, key: str, value: Any) -> None:
        """Store or update a semantic fact about the device or investigation."""
        with self._facts_lock():
            self._facts = self._load_facts()
            self._facts[key] = value
            history = self.memory_dir / f"fact_{self.run_id}_{uuid.uuid4().hex}.json"
            self._files.write_text(
                history,
                json.dumps(
                    {
                        "goal_id": self.goal_id,
                        "run_id": self.run_id,
                        "key": key,
                        "value": value,
                        "timestamp": time.time(),
                    },
                ),
                exclusive=True,
            )
            self._write_facts()

    def get_facts(self) -> dict[str, Any]:
        return dict(self._load_facts())

    def persist_turn(self, turn: TurnRecord) -> None:
        """Save turn record into episodic memory."""
        turn_file = (
            self.memory_dir / f"turn_{self.run_id}_{turn.turn_index:04d}_{uuid.uuid4().hex}.json"
        )
        data = {**turn.to_dict(), "goal_id": self.goal_id, "run_id": self.run_id}
        self._files.write_text(turn_file, json.dumps(data, indent=2), exclusive=True)

    def persist_result(self, result: AgentRunResult) -> None:
        path = self.memory_dir / f"result_{self.run_id}_{uuid.uuid4().hex}.json"
        self._files.write_text(
            path, json.dumps({**result.to_dict(), "run_id": self.run_id}, indent=2), exclusive=True
        )

    def load_episodic_turns(self) -> list[dict[str, Any]]:
        """Load all persisted turns from the case memory."""
        turns: list[dict[str, Any]] = []
        for name in sorted(self._files.names(".agent/memory")):
            if not (name.startswith("turn_") and name.endswith(".json")):
                continue
            try:
                turns.append(json.loads(self._files.read_text(self.memory_dir / name)))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return sorted(turns, key=lambda turn: turn.get("timestamp", 0))

    def truncate_observation_payload(self, tool_id: str, payload: Any) -> tuple[Any, str | None]:
        """Truncate large tool outputs, writing full output to an artifact file."""
        text = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
        if len(text) <= _MAX_OBSERVATION_CHARS:
            return payload, None

        # Content exceeds max chars - spill to disk
        timestamp = uuid.uuid4().hex
        safe_tool = re.sub(r"[^A-Za-z0-9_-]", "_", tool_id)[:64]
        artifact_path = self.artifacts_dir / f"obs_{safe_tool}_{timestamp}.txt"
        self._files.write_text(artifact_path, text, exclusive=True)

        preview = (
            text[:_MAX_OBSERVATION_CHARS]
            + f"\n\n[... Truncated {len(text) - _MAX_OBSERVATION_CHARS} chars. "
            + f"Full output saved to: {artifact_path}]"
        )
        return preview, str(artifact_path)

    def build_llm_messages(
        self,
        working_memory: WorkingMemory,
        consolidated_summary: str | None = None,
    ) -> list[dict[str, Any]]:
        """Construct the prompt messages array for the LLM with auto-compaction."""
        messages: list[dict[str, Any]] = []

        # 1. System/Goal message
        facts = self.get_facts()
        facts_summary = json.dumps(facts) if facts else "None recorded yet."
        scratchpad_text = "\n".join(f"- {note}" for note in working_memory.scratchpad) or "None."

        context_header = (
            f"=== MISSION OBJECTIVE ===\n{working_memory.goal.objective}\n\n"
            f"Target Device: {working_memory.goal.target_device or 'Auto-detect'}\n"
            f"Case Directory: {self.case_dir}\n\n"
            f"=== KNOWN SEMANTIC FACTS ===\n{facts_summary}\n\n"
            f"=== WORKING SCRATCHPAD ===\n{scratchpad_text}\n"
        )
        if consolidated_summary:
            context_header += (
                f"\n=== PRIOR CONSOLIDATED INVESTIGATION SUMMARY ===\n{consolidated_summary}\n"
            )

        messages.append({"role": "user", "content": context_header})

        # 2. Append turns
        for turn in working_memory.recent_turns:
            if turn.user_input is not None:
                messages.append({"role": "user", "content": turn.user_input})
            if turn.decision.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.decision.reasoning or turn.decision.text or "",
                        "tool_calls": [
                            {
                                "id": tc.call_id,
                                "type": "function",
                                "function": {
                                    "name": tc.tool_id,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in turn.decision.tool_calls
                        ],
                    }
                )
            elif turn.decision.text or turn.decision.reasoning:
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.decision.text or turn.decision.reasoning or "",
                    }
                )

            for obs in turn.observations:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": obs.call_id,
                        "content": json.dumps(obs.to_dict(), default=str),
                    }
                )

        return messages

    def check_and_compact(
        self,
        working_memory: WorkingMemory,
        consolidated_summary: str | None,
    ) -> tuple[WorkingMemory, str | None]:
        """Perform AutoCompact if estimated tokens exceed max_context_tokens."""
        messages = self.build_llm_messages(working_memory, consolidated_summary)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens = total_chars // _CHARS_PER_TOKEN

        if estimated_tokens <= self.max_context_tokens or len(working_memory.recent_turns) <= 4:
            return working_memory, consolidated_summary

        # AutoCompact: Keep the last 3 turns, compress older turns into bulleted summary
        older_turns = working_memory.recent_turns[:-3]
        preserved_turns = working_memory.recent_turns[-3:]

        summary_bullets: list[str] = []
        if consolidated_summary:
            summary_bullets.append(consolidated_summary)

        for t in older_turns:
            t_summary = f"- Turn {t.turn_index}: "
            if t.decision.tool_calls:
                t_tools = ", ".join(tc.tool_id for tc in t.decision.tool_calls)
                obs_status = "succeeded" if all(o.success for o in t.observations) else "had errors"
                t_summary += f"Executed [{t_tools}] -> {obs_status}."
            elif t.decision.text:
                t_summary += f"Reasoned: {t.decision.text[:120]}..."
            summary_bullets.append(t_summary)

        new_summary = "\n".join(summary_bullets)
        working_memory.recent_turns = preserved_turns
        logger.info(
            "AutoCompact triggered: compacted %d turns into consolidated summary",
            len(older_turns),
        )
        return working_memory, new_summary
