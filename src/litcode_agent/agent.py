"""完整的本地编程 Agent 循环与多轮会话。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from litcode_agent.hooks import HookEvent, HookExecution, HookOutcome, HookRunner
from litcode_agent.model import (
    AssistantTurn,
    Message,
    Model,
    ModelDelta,
    ModelError,
    ToolCall,
)
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
    "completed",
    "empty_response",
    "model_incomplete",
    "max_iterations",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: Literal[
        "model_start",
        "model_delta",
        "model_end",
        "tool_start",
        "tool_result",
        "hook_result",
    ]
    iteration: int
    tool_call: ToolCall | None = None
    hook_execution: HookExecution | None = None
    content: str | None = None
    is_error: bool = False
    has_tool_calls: bool = False


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

    def start_session(self) -> AgentSession:
        """创建一段保留消息历史的交互会话。"""

        return AgentSession(self)

    def run(self, task: str) -> AgentResult:
        """执行一次任务，并在返回前关闭对应会话。"""

        session = self.start_session()
        try:
            result = session.ask(task)
        except ModelError as error:
            session.close("model_error", str(error))
            raise
        session.close(result.reason, result.output)
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


class AgentSession:
    """一段只触发一次 SessionStart/SessionEnd 的多轮对话。"""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.session_id = str(uuid.uuid4())
        self.messages: list[Message] = [
            {"role": "system", "content": agent.system_prompt}
        ]
        self.started = False
        self.closed = False

    def ask(
        self,
        task: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentResult:
        if self.closed:
            raise RuntimeError("session is closed")
        if not task.strip():
            raise ValueError("task must not be empty")
        task = task.strip()
        if not self.started:
            self.started = True
            self.agent._run_hooks(
                "SessionStart",
                {
                    "session_id": self.session_id,
                    "cwd": str(self.agent._workspace()),
                    "hook_event_name": "SessionStart",
                    "source": "startup",
                    "task": task,
                },
                iteration=0,
                match_value="startup",
            )
        self.messages.append({"role": "user", "content": task})
        cancelled = should_cancel or (lambda: False)

        for iteration in range(1, self.agent.max_iterations + 1):
            if cancelled():
                return self._cancelled(iteration - 1)
            self.agent.event_sink(
                AgentEvent(kind="model_start", iteration=iteration)
            )
            turn, stream_cancelled = self._request_model(
                iteration,
                cancelled,
            )
            if stream_cancelled or cancelled():
                return self._cancelled(iteration)
            self.messages.append(turn.as_message())
            if not turn.tool_calls:
                if turn.finish_reason in {"length", "content_filter"}:
                    return self._result(
                        output=(
                            "The model response was incomplete "
                            f"(finish_reason={turn.finish_reason})."
                        ),
                        reason="model_incomplete",
                        iteration=iteration,
                    )
                if turn.content and turn.content.strip():
                    return self._result(
                        turn.content.strip(), "completed", iteration
                    )
                return self._result(
                    "The model returned neither text nor tool calls.",
                    "empty_response",
                    iteration,
                )

            for tool_call in turn.tool_calls:
                if cancelled():
                    return self._cancelled(iteration)
                self._execute_tool(tool_call, iteration)

        return self._result(
            f"Stopped after reaching the {self.agent.max_iterations}-iteration limit.",
            "max_iterations",
            self.agent.max_iterations,
        )

    def _request_model(
        self,
        iteration: int,
        cancelled: Callable[[], bool],
    ) -> tuple[AssistantTurn, bool]:
        stream = getattr(self.agent.model, "stream", None)
        if not callable(stream):
            turn = self.agent.model.complete(
                self.messages, self.agent.tools.schemas()
            )
            self.agent.event_sink(
                AgentEvent(
                    kind="model_end",
                    iteration=iteration,
                    content=turn.content,
                    has_tool_calls=bool(turn.tool_calls),
                )
            )
            return turn, False

        content_parts: list[str] = []
        tools: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        pending_text: list[str] = []
        last_flush = 0.0
        stream_cancelled = False
        deltas = stream(self.messages, self.agent.tools.schemas())
        try:
            for delta in deltas:
                if cancelled():
                    stream_cancelled = True
                    break
                if not isinstance(delta, ModelDelta):
                    raise ModelError("model stream returned an invalid delta")
                if delta.content:
                    content_parts.append(delta.content)
                    pending_text.append(delta.content)
                if delta.tool_index is not None:
                    tool = tools.setdefault(
                        delta.tool_index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    tool["id"] += delta.tool_call_id
                    tool["name"] += delta.tool_name
                    tool["arguments"] += delta.tool_arguments
                if delta.finish_reason is not None:
                    finish_reason = delta.finish_reason
                if pending_text and time.monotonic() - last_flush >= 0.04:
                    self._emit_delta(iteration, "".join(pending_text))
                    pending_text.clear()
                    last_flush = time.monotonic()
        finally:
            close = getattr(deltas, "close", None)
            if callable(close):
                close()
        if pending_text:
            self._emit_delta(iteration, "".join(pending_text))

        tool_calls = tuple(
            ToolCall(
                value["id"] or f"tool-{index}",
                value["name"],
                value["arguments"],
            )
            for index, value in sorted(tools.items())
        )
        turn = AssistantTurn(
            "".join(content_parts) or None,
            tool_calls,
            finish_reason,
        )
        self.agent.event_sink(
            AgentEvent(
                kind="model_end",
                iteration=iteration,
                content=turn.content,
                has_tool_calls=bool(tool_calls),
            )
        )
        return turn, stream_cancelled

    def _emit_delta(self, iteration: int, content: str) -> None:
        self.agent.event_sink(
            AgentEvent(
                kind="model_delta",
                iteration=iteration,
                content=content,
            )
        )

    def _cancelled(self, iteration: int) -> AgentResult:
        message = "本轮任务已由用户停止。"
        self.messages.append({"role": "assistant", "content": message})
        return self._result(
            message,
            "cancelled",
            iteration,
        )

    def close(self, reason: str = "user_exit", output: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        if not self.started:
            return
        self.agent._run_hooks(
            "SessionEnd",
            {
                "session_id": self.session_id,
                "cwd": str(self.agent._workspace()),
                "hook_event_name": "SessionEnd",
                "reason": reason,
                "output": output,
            },
            iteration=0,
            match_value=reason,
        )

    def _execute_tool(self, tool_call: ToolCall, iteration: int) -> None:
        self.agent.event_sink(
            AgentEvent(
                kind="tool_start",
                iteration=iteration,
                tool_call=tool_call,
            )
        )
        tool_input = _hook_tool_input(tool_call.arguments)
        hook_payload = {
            "session_id": self.session_id,
            "cwd": str(self.agent._workspace()),
            "hook_event_name": "PreToolUse",
            "tool_name": tool_call.name,
            "tool_input": tool_input,
            "tool_use_id": tool_call.id,
        }
        pre_tool = self.agent._run_hooks(
            "PreToolUse",
            hook_payload,
            iteration=iteration,
            match_value=tool_call.name,
        )
        result = (
            ToolResult.error(f"blocked by PreToolUse hook: {pre_tool.reason}")
            if pre_tool.blocked
            else self.agent.tools.execute_json(
                tool_call.name, tool_call.arguments
            )
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    {"ok": not result.is_error, "content": result.content},
                    ensure_ascii=False,
                ),
            }
        )
        post_event: HookEvent = (
            "PostToolUseFailure" if result.is_error else "PostToolUse"
        )
        self.agent._run_hooks(
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
        self.agent.event_sink(
            AgentEvent(
                kind="tool_result",
                iteration=iteration,
                tool_call=tool_call,
                content=result.content,
                is_error=result.is_error,
            )
        )

    def _result(
        self,
        output: str,
        reason: TerminationReason,
        iteration: int,
    ) -> AgentResult:
        return AgentResult(
            output=output,
            reason=reason,
            iterations=iteration,
            messages=tuple(self.messages),
        )


def _hook_tool_input(raw_arguments: str) -> object:
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": raw_arguments}
