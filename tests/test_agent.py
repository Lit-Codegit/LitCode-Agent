from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from litcode_agent.agent import Agent, AgentEvent
from litcode_agent.model import (
    AssistantTurn,
    Message,
    ModelDelta,
    ToolCall,
    ToolSchema,
)
from litcode_agent.tools.base import ToolResult
from litcode_agent.tools.registry import ToolRegistry
from litcode_agent.session_store import SessionStore
from litcode_agent.tools import ApplyPatchTool, Workspace


class FakeModel:
    def __init__(self, turns: Sequence[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.requests: list[list[Message]] = []

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AssistantTurn:
        self.requests.append(list(messages))
        return self.turns.pop(0)


class EchoTool:
    name = "echo"
    description = "Return the supplied text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, arguments: Mapping[str, object]) -> ToolResult:
        text = arguments.get("text")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return ToolResult(text)


def test_agent_returns_direct_model_answer() -> None:
    model = FakeModel([AssistantTurn("Done")])

    result = Agent(model, ToolRegistry([]), 3).run("Fix it")

    assert result.succeeded
    assert result.output == "Done"
    assert result.iterations == 1
    assert model.requests[0][-1] == {"role": "user", "content": "Fix it"}


def test_session_keeps_context_across_user_turns() -> None:
    model = FakeModel([AssistantTurn("第一次回答"), AssistantTurn("第二次回答")])
    session = Agent(model, ToolRegistry([]), 3).start_session()

    first = session.ask("第一个问题")
    second = session.ask("继续说明")
    session.close()

    assert first.output == "第一次回答"
    assert second.output == "第二次回答"
    assert model.requests[1][0]["role"] == "system"
    budget = model.requests[1][1]
    assert budget["role"] == "system"
    assert "litcode_iteration_budget" in budget["content"]
    assert "已用 0 / 上限 3，剩余 3" in budget["content"]
    assert model.requests[1][-2] == {
        "role": "assistant",
        "content": "第一次回答",
    }
    assert model.requests[1][-1] == {"role": "user", "content": "继续说明"}


def test_closed_session_rejects_new_turn() -> None:
    session = Agent(
        FakeModel([AssistantTurn("完成")]), ToolRegistry([]), 1
    ).start_session()
    session.ask("任务")
    session.close()

    with pytest.raises(RuntimeError, match="closed"):
        session.ask("继续")


def test_session_stops_at_cooperative_cancellation_boundary() -> None:
    cancelled = False

    class CancellingModel:
        def complete(self, messages, tools):
            nonlocal cancelled
            cancelled = True
            return AssistantTurn("不应显示的回答")

    session = Agent(CancellingModel(), ToolRegistry([]), 2).start_session()

    result = session.ask("开始", lambda: cancelled)

    assert result.reason == "cancelled"
    assert result.output == "本轮任务已由用户停止。"
    assert result.messages[-1] == {
        "role": "assistant",
        "content": "本轮任务已由用户停止。",
    }


def test_agent_executes_tool_and_links_result_to_call() -> None:
    model = FakeModel(
        [
            AssistantTurn(
                None,
                (ToolCall("call-1", "echo", '{"text":"hello"}'),),
            ),
            AssistantTurn("Finished"),
        ]
    )
    events: list[AgentEvent] = []

    result = Agent(model, ToolRegistry([EchoTool()]), 3, events.append).run("Go")

    tool_message = model.requests[1][-1]
    assert result.succeeded
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-1"
    assert json.loads(tool_message["content"]) == {  # type: ignore[arg-type]
        "ok": True,
        "content": "hello",
    }
    assert [event.kind for event in events] == [
        "model_start",
        "model_end",
        "tool_start",
        "tool_result",
        "model_start",
        "model_end",
    ]


def test_agent_returns_malformed_arguments_to_model() -> None:
    model = FakeModel(
        [
            AssistantTurn(None, (ToolCall("bad", "echo", "{"),)),
            AssistantTurn("Recovered"),
        ]
    )

    result = Agent(model, ToolRegistry([EchoTool()]), 3).run("Go")

    tool_result = json.loads(model.requests[1][-1]["content"])
    assert tool_result["ok"] is False
    assert "invalid JSON" in tool_result["content"]
    assert result.output == "Recovered"


def test_agent_stops_at_iteration_limit() -> None:
    model = FakeModel(
        [
            AssistantTurn(None, (ToolCall(str(index), "echo", '{"text":"x"}'),))
            for index in range(2)
        ]
        + [AssistantTurn("收尾结论")]
    )

    result = Agent(model, ToolRegistry([EchoTool()]), 2).run("Keep going")

    assert result.reason == "max_iterations"
    assert result.output == "收尾结论"
    assert result.iterations == 2
    assert len(model.requests) == 3
    sealed = model.requests[-1][1]["content"]
    assert "收尾轮" in sealed
    assert "不要调用工具" in sealed


def test_agent_wrap_up_ignores_tool_calls_and_keeps_history_consistent() -> None:
    model = FakeModel(
        [
            AssistantTurn(None, (ToolCall("a", "echo", '{"text":"x"}'),)),
            AssistantTurn(None, (ToolCall("b", "echo", '{"text":"y"}'),)),
            AssistantTurn(None, (ToolCall("late", "echo", '{"text":"z"}'),)),
        ]
    )

    result = Agent(model, ToolRegistry([EchoTool()]), 2).run("Keep going")

    assert result.reason == "max_iterations"
    assert "没有返回收尾内容" in result.output
    assert model.requests[-1][1]["content"].startswith("<litcode_iteration_budget>")
    assert result.messages[-1] == {
        "role": "assistant",
        "content": result.output,
    }


def test_agent_warns_when_budget_is_nearly_exhausted() -> None:
    model = FakeModel(
        [
            AssistantTurn(None, (ToolCall(str(index), "echo", '{"text":"x"}'),))
            for index in range(4)
        ]
        + [AssistantTurn("收尾")]
    )

    result = Agent(model, ToolRegistry([EchoTool()]), 4).run("Keep going")

    assert result.reason == "max_iterations"
    assert result.output == "收尾"
    first_round = next(
        message["content"]
        for message in model.requests[0]
        if message.get("role") == "system"
        and "litcode_iteration_budget" in message["content"]
    )
    assert "已用 0 / 上限 4，剩余 4" in first_round
    third_round = next(
        message["content"]
        for message in model.requests[2]
        if message.get("role") == "system"
        and "litcode_iteration_budget" in message["content"]
    )
    assert "已用 2 / 上限 4，剩余 2" in third_round
    assert any(
        "预算接近耗尽" in message["content"]
        for message in model.requests[3]
        if message.get("role") == "system"
    )


def test_agent_reports_empty_model_response() -> None:
    result = Agent(
        FakeModel([AssistantTurn(None)]), ToolRegistry([]), 2
    ).run("Go")

    assert result.reason == "empty_response"
    assert not result.succeeded


def test_agent_accumulates_streamed_text_and_tool_arguments() -> None:
    class StreamingModel:
        def __init__(self) -> None:
            self.requests = []

        def stream(self, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                yield ModelDelta(
                    tool_index=0,
                    tool_call_id="stream-call",
                    tool_name="echo",
                    tool_arguments='{"text":',
                )
                yield ModelDelta(
                    tool_index=0,
                    tool_arguments='"流式"}',
                    finish_reason="tool_calls",
                )
            else:
                yield ModelDelta(content="完")
                yield ModelDelta(content="成", finish_reason="stop")

    model = StreamingModel()
    events: list[AgentEvent] = []

    result = Agent(
        model, ToolRegistry([EchoTool()]), 3, events.append
    ).run("执行")

    assert result.output == "完成"
    assert json.loads(model.requests[1][-1]["content"])["content"] == "流式"
    assert [event.content for event in events if event.kind == "model_delta"] == [
        "完",
        "成",
    ]
    assert [event.kind for event in events].count("model_end") == 2


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_agent_does_not_treat_incomplete_response_as_success(
    finish_reason: str,
) -> None:
    result = Agent(
        FakeModel([AssistantTurn("partial answer", finish_reason=finish_reason)]),
        ToolRegistry([]),
        2,
    ).run("Go")

    assert result.reason == "model_incomplete"
    assert not result.succeeded


def test_session_compaction_keeps_raw_history_and_changes_model_view(
    tmp_path,
) -> None:
    model = FakeModel(
        [
            AssistantTurn("第一次回答"),
            AssistantTurn("压缩摘要"),
            AssistantTurn("继续回答"),
        ]
    )
    store = SessionStore(tmp_path / "sessions.db")
    agent = Agent(
        model,
        ToolRegistry([]),
        3,
        store=store,
        model_name="model",
        workspace=tmp_path,
    )
    session = agent.start_session()
    session.ask("第一个问题")

    summary = session.compact()
    session.ask("继续")

    assert summary == "压缩摘要"
    assert store.load(session.session_id)[1]["content"] == "第一个问题"
    assert any(
        "受信压缩摘要" in message["content"]
        for message in model.requests[-1]
        if message.get("role") == "user"
    )
    assert model.requests[-1][-1]["content"] == "继续"


def test_rewind_can_restore_agent_edits_and_redo_them(tmp_path) -> None:
    model = FakeModel(
        [
            AssistantTurn(
                None,
                (ToolCall("create", "apply_patch", '{"path":"a.txt","old_text":"","new_text":"one"}'),),
            ),
            AssistantTurn("第一轮完成"),
            AssistantTurn(
                None,
                (ToolCall("edit", "apply_patch", '{"path":"a.txt","old_text":"one","new_text":"two"}'),),
            ),
            AssistantTurn("第二轮完成"),
        ]
    )
    store = SessionStore(tmp_path / "sessions.db")
    session = Agent(
        model,
        ToolRegistry([ApplyPatchTool(Workspace(tmp_path))]),
        3,
        store=store,
        model_name="model",
        workspace=tmp_path,
    ).start_session()
    session.ask("第一轮")
    first = session.checkpoints()[0]
    session.ask("第二轮")
    assert (tmp_path / "a.txt").read_text() == "two"

    restored = session.rewind(first, restore_files=True)
    assert restored == 1
    assert (tmp_path / "a.txt").read_text() == "one"

    assert session.redo() == 1
    assert (tmp_path / "a.txt").read_text() == "two"
