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
from litcode_agent.tools.base import FileChange, ToolResult
from litcode_agent.tools.base import ToolExecutionContext, UserDeclinedError
from litcode_agent.tools.registry import ToolRegistry
from litcode_agent.session_store import Checkpoint, SessionStore
from litcode_agent.tools.workspace import Workspace

SYSTEM_PROMPT = """You are LitCode Agent, a careful coding assistant operating in a local workspace.

Use the provided tools to inspect the repository before editing it. Make the
smallest changes that fully solve the user's task. Read relevant files, apply
precise edits, and run appropriate verification. Treat tool errors as feedback:
correct the request and try again. Never claim that a command or test succeeded
unless its tool result says so. When the task is complete, respond with a concise
summary of changes, tests run, and any remaining limitations.

When a genuine user decision shapes the direction of your work, ask the user
with the ask_user tool instead of guessing: clarify ambiguous instructions,
gather preferences, or get an explicit choice between approaches. Wait for the
answer before acting on it. Ask the minimum number of questions and prefer
providing a recommended option first with "(Recommended)" appended to its label.
Do not use the tool for trivia you can resolve yourself; if the user rejects a
question, adapt by choosing the best default and continue.

User decisions are final. When the user declines a confirmation, rejects an
action, or changes the direction of the task, that decision stands: do not
retry the same operation through a different command, script, or tool, do not
argue, and do not seek another way to reach the rejected outcome. Stop, report
the decision, and wait for new instructions. Ask before acting whenever an
action was previously declined.
"""

TerminationReason = Literal[
    "completed",
    "empty_response",
    "model_incomplete",
    "max_iterations",
    "cancelled",
    "user_declined",
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
    file_change: FileChange | None = None
    has_tool_calls: bool = False
    session_id: str | None = None


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
        store: SessionStore | None = None,
        model_name: str = "unknown",
        workspace: Path | None = None,
        runtime_context: Callable[[], str] | None = None,
        tool_context: Callable[[str], ToolExecutionContext] | None = None,
        origin_terminal_id: str | None = None,
        origin_pane_slot: int | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.event_sink = event_sink or (lambda event: None)
        self.hooks = hooks
        self.system_prompt = system_prompt
        self.store = store
        self.model_name = model_name
        self.workspace = workspace.resolve() if workspace is not None else None
        self.runtime_context = runtime_context
        self.tool_context = tool_context
        self.origin_terminal_id = origin_terminal_id
        self.origin_pane_slot = origin_pane_slot

    def start_session(self, session_id: str | None = None) -> AgentSession:
        """创建一段保留消息历史的交互会话。"""

        return AgentSession(self, session_id)

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
                    session_id=(
                        str(payload["session_id"])
                        if "session_id" in payload
                        else None
                    ),
                )
            )
        return outcome

    def _workspace(self) -> Path:
        if self.workspace is not None:
            return self.workspace
        return self.hooks.workspace if self.hooks is not None else Path.cwd()


class AgentSession:
    """一段只触发一次 SessionStart/SessionEnd 的多轮对话。"""

    def __init__(self, agent: Agent, session_id: str | None = None) -> None:
        self.agent = agent
        initial: list[Message] = [{"role": "system", "content": agent.system_prompt}]
        if session_id is not None and agent.store is not None:
            self.session_id = session_id
            self.messages = list(agent.store.load(session_id))
        else:
            self.session_id = str(uuid.uuid4())
            self.messages = initial
            if agent.store is not None:
                self.session_id = agent.store.create(
                    agent._workspace(), agent.model_name, self.messages,
                    session_id=self.session_id,
                    origin_terminal_id=agent.origin_terminal_id,
                    origin_pane_slot=agent.origin_pane_slot,
                )
        saved_summary = agent.store.summary(self.session_id) if agent.store else None
        self.summary = saved_summary
        self._redo: (
            tuple[list[Message], Checkpoint, bool, tuple[str, int] | None] | None
        ) = None
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
        if self._redo is not None:
            _, checkpoint, restored_files, _ = self._redo
            if restored_files and self.agent.store is not None:
                self.agent.store.discard_changes_after(
                    self.session_id, checkpoint.file_cursor
                )
            if self.agent.store is not None:
                self.agent.store.discard_checkpoints_after(
                    self.session_id, checkpoint.created_at
                )
            self._redo = None
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
        self._append_message({"role": "user", "content": task})
        cancelled = should_cancel or (lambda: False)

        for iteration in range(1, self.agent.max_iterations + 1):
            if cancelled():
                return self._cancelled(iteration - 1)
            self.agent.event_sink(
                AgentEvent(
                    kind="model_start",
                    iteration=iteration,
                    session_id=self.session_id,
                )
            )
            turn, stream_cancelled = self._request_model(
                iteration,
                cancelled,
            )
            if stream_cancelled or cancelled():
                return self._cancelled(iteration)
            self._append_message(turn.as_message())
            if not turn.tool_calls:
                if turn.finish_reason in {"length", "content_filter"}:
                    return self._finish_turn(task, self._result(
                        output=(
                            "The model response was incomplete "
                            f"(finish_reason={turn.finish_reason})."
                        ),
                        reason="model_incomplete",
                        iteration=iteration,
                    ))
                if turn.content and turn.content.strip():
                    return self._finish_turn(task, self._result(
                        turn.content.strip(), "completed", iteration
                    ))
                return self._finish_turn(task, self._result(
                    "The model returned neither text nor tool calls.",
                    "empty_response",
                    iteration,
                ))

            for tool_call in turn.tool_calls:
                if cancelled():
                    return self._cancelled(iteration)
                try:
                    self._execute_tool(tool_call, iteration)
                except UserDeclinedError as error:
                    return self._finish_turn(task, self._result(
                        str(error),
                        "user_declined",
                        iteration,
                    ))

        # Budget exhausted while the model was still working: one sealed
        # final round lets it deliver a wrap-up instead of stopping silent.
        return self._finish_turn(task, self._wrap_up(cancelled))

    def _wrap_up(self, cancelled: Callable[[], bool]) -> AgentResult:
        if cancelled():
            return self._cancelled(self.agent.max_iterations)
        iteration = self.agent.max_iterations + 1
        self.agent.event_sink(
            AgentEvent(
                kind="model_start",
                iteration=iteration,
                session_id=self.session_id,
            )
        )
        turn, stream_cancelled = self._request_model(iteration, cancelled, sealed=True)
        if stream_cancelled or cancelled():
            return self._cancelled(iteration)
        content = (turn.content or "").strip() or (
            "迭代预算已耗尽，但模型没有返回收尾内容；"
            "可继续追问或使用 /compact 压缩上下文。"
        )
        self._append_message({"role": "assistant", "content": content})
        return self._result(content, "max_iterations", self.agent.max_iterations)

    def _request_model(
        self,
        iteration: int,
        cancelled: Callable[[], bool],
        *,
        sealed: bool = False,
    ) -> tuple[AssistantTurn, bool]:
        stream = getattr(self.agent.model, "stream", None)
        if not callable(stream):
            turn = self.agent.model.complete(
                self._model_messages(iteration, sealed=sealed), self.agent.tools.schemas()
            )
            self.agent.event_sink(
                AgentEvent(
                    kind="model_end",
                    iteration=iteration,
                    content=turn.content,
                    has_tool_calls=bool(turn.tool_calls),
                    session_id=self.session_id,
                )
            )
            return turn, False

        content_parts: list[str] = []
        tools: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        pending_text: list[str] = []
        last_flush = 0.0
        stream_cancelled = False
        deltas = stream(self._model_messages(iteration, sealed=sealed), self.agent.tools.schemas())
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
                session_id=self.session_id,
            )
        )
        return turn, stream_cancelled

    def _emit_delta(self, iteration: int, content: str) -> None:
        self.agent.event_sink(
            AgentEvent(
                kind="model_delta",
                iteration=iteration,
                content=content,
                session_id=self.session_id,
            )
        )

    def _cancelled(self, iteration: int) -> AgentResult:
        message = "本轮任务已由用户停止。"
        self._append_message({"role": "assistant", "content": message})
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
                session_id=self.session_id,
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
        try:
            result = (
                ToolResult.error(f"blocked by PreToolUse hook: {pre_tool.reason}")
                if pre_tool.blocked
                else self.agent.tools.execute_json(
                    tool_call.name,
                    tool_call.arguments,
                    (
                        self.agent.tool_context(self.session_id)
                        if self.agent.tool_context is not None
                        else ToolExecutionContext(
                            self.session_id, self.agent._workspace()
                        )
                    ),
                )
            )
        except UserDeclinedError as error:
            self._append_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {
                            "ok": False,
                            "declined": True,
                            "content": str(error),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            self.agent.event_sink(
                AgentEvent(
                    kind="tool_result",
                    iteration=iteration,
                    tool_call=tool_call,
                    content=str(error),
                    is_error=True,
                    session_id=self.session_id,
                )
            )
            raise
        self._append_message(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    {"ok": not result.is_error, "content": result.content},
                    ensure_ascii=False,
                ),
            }
        )
        if result.file_change is not None and self.agent.store is not None:
            self.agent.store.record_change(self.session_id, result.file_change)
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
                file_change=result.file_change,
                session_id=self.session_id,
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

    def _append_message(self, message: Message) -> None:
        self.messages.append(message)
        if self.agent.store is not None:
            self.agent.store.save_messages(self.session_id, self.messages)

    def _finish_turn(self, task: str, result: AgentResult) -> AgentResult:
        if self.agent.store is not None:
            title = task.replace("\n", " ")[:48]
            self.agent.store.save_messages(self.session_id, self.messages, title=title)
            self.agent.store.add_checkpoint(self.session_id, title, self.messages)
        return result

    def _model_messages(
        self, iteration: int | None = None, *, sealed: bool = False
    ) -> list[Message]:
        if self.summary is None:
            messages = list(self.messages)
        else:
            summary, boundary = self.summary
            messages = [
                self.messages[0],
                {
                    "role": "user",
                    "content": "以下是此前会话的受信压缩摘要：\n\n" + summary,
                },
                *self.messages[boundary + 1 :],
            ]
        budget = self._budget_note(iteration, sealed=sealed) if iteration else None
        if self.agent.runtime_context is None:
            if budget is None:
                return messages
            return [
                messages[0],
                {"role": "system", "content": budget},
                *messages[1:],
            ]
        runtime = self.agent.runtime_context()
        blocks = [
            (
                "<litcode_runtime_location>\n"
                f"{runtime}\n"
                "</litcode_runtime_location>"
            )
        ]
        if budget is not None:
            blocks.append(budget)
        return [
            messages[0],
            {"role": "system", "content": "\n".join(blocks)},
            *messages[1:],
        ]

    def _budget_note(self, iteration: int, *, sealed: bool) -> str:
        used = max(0, iteration - 1)
        remaining = max(0, self.agent.max_iterations - used)
        if sealed:
            return (
                "<litcode_iteration_budget>\n"
                f"预算已耗尽（已用 {used} 轮）。这是收尾轮：不要调用工具，"
                "直接用已有信息写出最终交付或摘要，然后结束。\n"
                "</litcode_iteration_budget>"
            )
        note = (
            f"<litcode_iteration_budget>已用 {used} / 上限 {remaining + used}，"
            f"剩余 {remaining}</litcode_iteration_budget>"
        )
        if remaining <= max(6, self.agent.max_iterations // 4):
            note += (
                "\n预算接近耗尽：立即停止探索与无关读取，优先产出最终结果；"
                "需要更多工时请委派子会话或明示预算不足。"
            )
        return note

    def compact(self, instructions: str = "") -> str:
        """Create a manual summary checkpoint without deleting raw history."""

        if len(self.messages) <= 1:
            raise ValueError("当前会话没有可压缩的内容")
        request: Message = {
            "role": "user",
            "content": (
                "请把以上会话压缩成可供后续模型继续工作的中文摘要。固定栏目："
                "用户约束、关键决定、已完成、未完成、相关文件、下一步。"
                "不要虚构事实。" + (f"\n额外要求：{instructions}" if instructions else "")
            ),
        }
        turn = self.agent.model.complete([*self._model_messages(), request], [])
        if not turn.content or not turn.content.strip():
            raise ModelError("上下文压缩没有返回摘要")
        boundary = len(self.messages) - 1
        summary = turn.content.strip()
        self.summary = (summary, boundary)
        if self.agent.store is not None:
            self.agent.store.save_summary(self.session_id, summary, boundary)
            self.agent.store.add_checkpoint(self.session_id, "上下文压缩", self.messages)
        return summary

    def checkpoints(self) -> tuple[Checkpoint, ...]:
        if self.agent.store is None:
            return ()
        return self.agent.store.checkpoints(self.session_id)

    def rewind(self, checkpoint: Checkpoint, *, restore_files: bool) -> int:
        if self.agent.store is None:
            raise RuntimeError("会话存储未启用")
        self._redo = (
            list(self.messages), checkpoint, restore_files, self.summary
        )
        restored = 0
        if restore_files:
            restored = self.agent.store.restore_files(
                self.session_id, checkpoint.file_cursor, Workspace(self.agent._workspace())
            )
        self.messages = list(checkpoint.messages)
        self.summary = None
        self.agent.store.clear_summary(self.session_id)
        self.agent.store.save_messages(self.session_id, self.messages)
        return restored

    def redo(self) -> int:
        if self.agent.store is None or self._redo is None:
            raise RuntimeError("没有可恢复的 rewind")
        messages, checkpoint, restore_files, summary = self._redo
        restored = 0
        if restore_files:
            restored = self.agent.store.restore_files(
                self.session_id,
                checkpoint.file_cursor,
                Workspace(self.agent._workspace()),
                forward=True,
            )
        self.messages = messages
        self.summary = summary
        self.agent.store.save_messages(self.session_id, self.messages)
        if summary is None:
            self.agent.store.clear_summary(self.session_id)
        else:
            self.agent.store.save_summary(self.session_id, *summary)
        self._redo = None
        return restored

    def fork(self, checkpoint: Checkpoint) -> AgentSession:
        if self.agent.store is None:
            raise RuntimeError("会话存储未启用")
        identifier = self.agent.store.fork(
            self.session_id, checkpoint, self.agent.model_name
        )
        return self.agent.start_session(identifier)


def _hook_tool_input(raw_arguments: str) -> object:
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": raw_arguments}
