"""The complete local coding-agent control loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from litcode_agent.model import Message, Model, ToolCall
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
    kind: Literal["model_start", "tool_start", "tool_result"]
    iteration: int
    tool_call: ToolCall | None = None
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
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.event_sink = event_sink or (lambda event: None)
        self.system_prompt = system_prompt

    def run(self, task: str) -> AgentResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.strip()},
        ]

        for iteration in range(1, self.max_iterations + 1):
            self.event_sink(AgentEvent(kind="model_start", iteration=iteration))
            turn = self.model.complete(messages, self.tools.schemas())
            messages.append(turn.as_message())
            if not turn.tool_calls:
                if turn.finish_reason in {"length", "content_filter"}:
                    return AgentResult(
                        output=(
                            "The model response was incomplete "
                            f"(finish_reason={turn.finish_reason})."
                        ),
                        reason="model_incomplete",
                        iterations=iteration,
                        messages=tuple(messages),
                    )
                if turn.content and turn.content.strip():
                    return AgentResult(
                        output=turn.content.strip(),
                        reason="completed",
                        iterations=iteration,
                        messages=tuple(messages),
                    )
                return AgentResult(
                    output="The model returned neither text nor tool calls.",
                    reason="empty_response",
                    iterations=iteration,
                    messages=tuple(messages),
                )

            for tool_call in turn.tool_calls:
                self.event_sink(
                    AgentEvent(
                        kind="tool_start",
                        iteration=iteration,
                        tool_call=tool_call,
                    )
                )
                result = self.tools.execute_json(
                    tool_call.name, tool_call.arguments
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
                self.event_sink(
                    AgentEvent(
                        kind="tool_result",
                        iteration=iteration,
                        tool_call=tool_call,
                        content=result.content,
                        is_error=result.is_error,
                    )
                )

        return AgentResult(
            output=f"Stopped after reaching the {self.max_iterations}-iteration limit.",
            reason="max_iterations",
            iterations=self.max_iterations,
            messages=tuple(messages),
        )
