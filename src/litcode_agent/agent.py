"""The complete local coding-agent control loop."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from litcode_agent.hooks import HookEvent, HookExecution, HookOutcome, HookRunner
from litcode_agent.model import Message, Model, ModelError, ToolCall
from litcode_agent.tools.base import ToolResult
from litcode_agent.tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are LitCode Agent, a careful coding assistant operating in a local workspace.

Use the provided tools to inspect the repository before editing it. Make the
smallest changes that fully solve the user's task. Read relevant files, apply
precise edits, and run appropriate verification. Treat tool errors as feedback:
correct the request and try again. Never claim that a command or test succeeded
unless its tool result says so. When the task is complete, respond with a concise
summary of changes, tests run, and any remaining limitations.
"""

TerminationReason = Literal[
    "completed", "empty_response", "model_incomplete", "max_iterations"
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: Literal["model_start", "tool_start", "tool_result", "hook_result"]
    iteration: int
    tool_call: ToolCall | None = None
    hook_execution: HookExecution | None = None
    content: str | None = None
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class AgentResult:
    output: str
    reason: TerminationReason
    iterations: int
    messages: tuple[Message, ...]

    @property
    def succeeded(self) -> bool:
        return self.reason == "completed"


EventSink = Callable[[AgentEvent], None]


class Agent:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        max_iterations: int,
        event_sink: EventSink | None = None,
        hooks: HookRunner | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.event_sink = event_sink or (lambda event: None)
        self.hooks = hooks
        self.system_prompt = system_prompt

    def run(self, task: str) -> AgentResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.strip()},
        ]
        session_id = str(uuid.uuid4())
        self._run_hooks(
            "SessionStart",
            {
                "session_id": session_id,
                "cwd": str(self._workspace()),
                "hook_event_name": "SessionStart",
                "source": "startup",
                "task": task.strip(),
            },
            iteration=0,
            match_value="startup",
        )

        for iteration in range(1, self.max_iterations + 1):
            self.event_sink(AgentEvent(kind="model_start", iteration=iteration))
            try:
                turn = self.model.complete(messages, self.tools.schemas())
            except ModelError as error:
                self._run_hooks(
                    "SessionEnd",
                    {
                        "session_id": session_id,
                        "cwd": str(self._workspace()),
                        "hook_event_name": "SessionEnd",
                        "reason": "model_error",
                        "output": str(error),
                    },
                    iteration=iteration,
                    match_value="model_error",
                )
                raise
            messages.append(turn.as_message())
            if not turn.tool_calls:
                if turn.finish_reason in {"length", "content_filter"}:
                    return self._finish(
                        AgentResult(
                            output=(
                                "The model response was incomplete "
                                f"(finish_reason={turn.finish_reason})."
                            ),
                            reason="model_incomplete",
                            iterations=iteration,
                            messages=tuple(messages),
                        ),
                        session_id,
                    )
                if turn.content and turn.content.strip():
                    return self._finish(
                        AgentResult(
                            output=turn.content.strip(),
                            reason="completed",
                            iterations=iteration,
                            messages=tuple(messages),
                        ),
                        session_id,
                    )
                return self._finish(
                    AgentResult(
                        output="The model returned neither text nor tool calls.",
                        reason="empty_response",
                        iterations=iteration,
                        messages=tuple(messages),
                    ),
                    session_id,
                )

            for tool_call in turn.tool_calls:
                self.event_sink(
                    AgentEvent(
                        kind="tool_start",
                        iteration=iteration,
                        tool_call=tool_call,
                    )
                )
                tool_input = _hook_tool_input(tool_call.arguments)
                hook_payload = {
                    "session_id": session_id,
                    "cwd": str(self._workspace()),
                    "hook_event_name": "PreToolUse",
                    "tool_name": tool_call.name,
                    "tool_input": tool_input,
                    "tool_use_id": tool_call.id,
                }
                pre_tool = self._run_hooks(
                    "PreToolUse",
                    hook_payload,
                    iteration=iteration,
                    match_value=tool_call.name,
                )
                result = (
                    ToolResult.error(
                        f"blocked by PreToolUse hook: {pre_tool.reason}"
                    )
                    if pre_tool.blocked
                    else self.tools.execute_json(
                        tool_call.name, tool_call.arguments
                    )
                )
                result_content = json.dumps(
                    {"ok": not result.is_error, "content": result.content},
                    ensure_ascii=False,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                    }
                )
                post_event: HookEvent = (
                    "PostToolUseFailure" if result.is_error else "PostToolUse"
                )
                self._run_hooks(
                    post_event,
                    {
                        **hook_payload,
                        "hook_event_name": post_event,
                        "tool_response": {
                            "ok": not result.is_error,
                            "content": result.content,
                        },
                    },
                    iteration=iteration,
                    match_value=tool_call.name,
                )
                self.event_sink(
                    AgentEvent(
                        kind="tool_result",
                        iteration=iteration,
                        tool_call=tool_call,
                        content=result.content,
                        is_error=result.is_error,
                    )
                )

        return self._finish(
            AgentResult(
                output=(
                    f"Stopped after reaching the {self.max_iterations}-iteration limit."
                ),
                reason="max_iterations",
                iterations=self.max_iterations,
                messages=tuple(messages),
            ),
            session_id,
        )

    def _finish(self, result: AgentResult, session_id: str) -> AgentResult:
        self._run_hooks(
            "SessionEnd",
            {
                "session_id": session_id,
                "cwd": str(self._workspace()),
                "hook_event_name": "SessionEnd",
                "reason": result.reason,
                "output": result.output,
            },
            iteration=result.iterations,
            match_value=result.reason,
        )
        return result

    def _run_hooks(
        self,
        event: HookEvent,
        payload: dict[str, object],
        *,
        iteration: int,
        match_value: str,
    ) -> HookOutcome:
        if self.hooks is None:
            return HookOutcome()
        outcome = self.hooks.run(event, payload, match_value=match_value)
        for execution in outcome.executions:
            self.event_sink(
                AgentEvent(
                    kind="hook_result",
                    iteration=iteration,
                    hook_execution=execution,
                )
            )
        return outcome

    def _workspace(self) -> Path:
        return self.hooks.workspace if self.hooks is not None else Path.cwd()


def _hook_tool_input(raw_arguments: str) -> object:
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": raw_arguments}
