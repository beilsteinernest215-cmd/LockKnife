from __future__ import annotations

import dataclasses
import logging
import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any

from lockknife.core.agent.memory import MemoryStore
from lockknife.core.agent.models import (
    AgentGoal,
    AgentRunResult,
    DecisionKind,
    GoalStatus,
    ModelDecision,
    ToolObservation,
    TurnRecord,
)
from lockknife.core.agent.planner import GoalPlan
from lockknife.core.agent.policy import ResearcherPolicy
from lockknife.core.agent.provider import (
    DeterministicMockProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    ProviderConfig,
)
from lockknife.core.agent.steering import SteeringQueue
from lockknife.core.agent.subagent import ExecutionBudget, SubagentManager
from lockknife.core.agent.tools import AgentToolRegistry
from lockknife.core.agent.turn import TurnEngine

logger = logging.getLogger("lockknife.agent.runtime")


class AutonomousRuntime:
    """The master runtime coordinating goal execution, memory, tools, and turn engine."""

    def __init__(
        self,
        *,
        goal: AgentGoal,
        case_dir: pathlib.Path | None = None,
        target_device: str | None = None,
        provider: LLMProvider | None = None,
        policy: ResearcherPolicy | None = None,
        plan: GoalPlan | None = None,
        steering: SteeringQueue | None = None,
        max_concurrency: int = 4,
        action_callback: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        on_turn_start: Callable[[int], None] | None = None,
        on_turn_decision: Callable[[TurnRecord], None] | None = None,
        on_tool_execute: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[ToolObservation], None] | None = None,
        execution_budget: ExecutionBudget | None = None,
        delegation_depth: int = 0,
    ) -> None:
        self.goal = goal
        self.policy = policy or ResearcherPolicy(unrestricted=True, auto_provision_case=True)
        self.case_dir = self.policy.ensure_case_workspace(case_dir)
        self.goal.case_dir = self.case_dir
        self.target_device = target_device or self.goal.target_device
        self.goal.target_device = self.target_device

        self.plan = plan or GoalPlan(goal)
        self.steering = steering or SteeringQueue()
        self.max_concurrency = max(1, max_concurrency)
        self.execution_budget = execution_budget or ExecutionBudget(goal.budget_iterations)
        self._chat_lock = threading.RLock()

        self.provider = provider or OpenAICompatibleProvider()
        self.memory = MemoryStore(case_dir=self.case_dir, goal_id=self.goal.goal_id)

        # Wire subagent manager with a factory targeting this runtime class
        def _subagent_runtime_factory(
            goal: AgentGoal, case_dir: pathlib.Path, target_device: str | None
        ) -> AutonomousRuntime:
            if delegation_depth >= self.execution_budget.max_depth:
                raise RuntimeError("Delegation depth limit reached")
            return AutonomousRuntime(
                goal=goal,
                case_dir=case_dir,
                target_device=target_device or self.target_device,
                provider=self.provider,
                policy=self.policy,
                action_callback=action_callback,
                max_concurrency=self.max_concurrency,
                execution_budget=self.execution_budget,
                delegation_depth=delegation_depth + 1,
            )

        self.subagent_manager = SubagentManager(
            parent_case_dir=self.case_dir,
            runtime_factory=_subagent_runtime_factory,
            parent_goal_id=self.goal.goal_id,
        )

        self.tools = AgentToolRegistry(
            case_dir=self.case_dir,
            target_serial=self.target_device,
            action_callback=action_callback,
            subagent_manager=self.subagent_manager,
            memory_store=self.memory,
        )

        self.turn_engine = TurnEngine(
            goal=self.goal,
            provider=self.provider,
            tools=self.tools,
            memory=self.memory,
            policy=self.policy,
            plan=self.plan,
            steering=self.steering,
            max_concurrency=self.max_concurrency,
            on_turn_start=on_turn_start,
            on_turn_decision=on_turn_decision,
            on_tool_execute=on_tool_execute,
            on_tool_result=on_tool_result,
            execution_budget=self.execution_budget,
        )

    def run(self) -> AgentRunResult:
        """Execute the goal to completion or terminal budget."""
        result = self.turn_engine.run_loop()
        self.memory.persist_result(result)
        return result

    def steer(self, guidance: str) -> None:
        """Inject mid-flight guidance into the running agent's steering queue."""
        self.steering.push(guidance)

    def chat_step(self, user_input: str) -> str:
        """Complete one user exchange, retaining history and the shared session budget."""
        with self._chat_lock:
            responses: list[str] = []
            turns: list[TurnRecord] = []
            artifacts: list[str] = []
            pending_input: str | None = user_input
            started = time.monotonic()
            self.goal.status = GoalStatus.ACTIVE
            try:
                while True:
                    engine = self.turn_engine
                    for note in self.steering.drain():
                        engine.working_memory.add_scratchpad_note(note)
                    engine.working_memory, engine.consolidated_summary = (
                        self.memory.check_and_compact(
                            engine.working_memory, engine.consolidated_summary
                        )
                    )
                    messages = self.memory.build_llm_messages(
                        engine.working_memory, engine.consolidated_summary
                    )
                    if pending_input is not None:
                        messages.append({"role": "user", "content": pending_input})
                    if not self.execution_budget.consume():
                        self.goal.status = GoalStatus.BUDGET_EXHAUSTED
                        decision = ModelDecision.failure(
                            "Aggregate agent decision budget exhausted"
                        )
                    else:
                        try:
                            decision = self.provider.complete(
                                messages=messages, tools=self.tools.get_tool_specs()
                            )
                        except Exception as exc:
                            decision = ModelDecision.failure(str(exc))

                    turn = TurnRecord(
                        turn_index=max(
                            (t.turn_index for t in engine.working_memory.recent_turns), default=0
                        )
                        + 1,
                        decision=decision,
                        user_input=pending_input,
                    )
                    pending_input = None
                    if decision.is_tool_call:
                        for call in decision.tool_calls:
                            responses.append(f"-> Calling tool: {call.tool_id}({call.arguments})")
                            auth = self.policy.authorize_tool(
                                call.tool_id, call.arguments, self.case_dir
                            )
                            observation = (
                                self.tools.execute(call)
                                if auth.allowed
                                else ToolObservation(
                                    call_id=call.call_id,
                                    tool_id=call.tool_id,
                                    success=False,
                                    error=f"Denied: {auth.reason}",
                                )
                            )
                            preview, artifact = self.memory.truncate_observation_payload(
                                call.tool_id, observation.output
                            )
                            if artifact:
                                artifacts.append(artifact)
                            artifacts.extend(observation.artifacts_created)
                            observation = dataclasses.replace(observation, output=preview)
                            turn.observations.append(observation)
                            responses.append(
                                f"   Result: {'OK' if observation.success else 'FAILED'}"
                            )
                            if observation.error:
                                responses.append(f"   Error: {observation.error}")
                    engine.working_memory.recent_turns.append(turn)
                    turns.append(turn)
                    self.memory.persist_turn(turn)
                    if decision.kind == DecisionKind.ERROR:
                        if self.goal.status != GoalStatus.BUDGET_EXHAUSTED:
                            self.goal.status = GoalStatus.FAILED
                        raise RuntimeError(decision.text)
                    if decision.is_terminal:
                        self.goal.status = GoalStatus.COMPLETED
                        responses.append(decision.text or "Acknowledged.")
                        return "\n".join(responses)
                    if not decision.is_tool_call and decision.kind != DecisionKind.CONTINUE:
                        self.goal.status = GoalStatus.FAILED
                        raise RuntimeError("Provider returned an empty tool decision")
            except Exception:
                if self.goal.status == GoalStatus.ACTIVE:
                    self.goal.status = GoalStatus.FAILED
                self.close()
                raise
            finally:
                self.memory.persist_result(
                    AgentRunResult(
                        goal=self.goal,
                        status=self.goal.status,
                        final_response="\n".join(responses),
                        turns=turns,
                        artifacts_created=artifacts,
                        duration_s=time.monotonic() - started,
                        error=turns[-1].decision.text
                        if turns
                        and self.goal.status in {GoalStatus.FAILED, GoalStatus.BUDGET_EXHAUSTED}
                        else None,
                    )
                )

    def close(self) -> None:
        self.tools.exec_sessions.close_all()
