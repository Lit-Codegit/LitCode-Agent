from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from litcode_agent.agent import Agent, AgentEvent
from litcode_agent.model import AssistantTurn, Message, ToolCall, ToolSchema
from litcode_agent.tools.base import ToolResult
from litcode_agent.tools.registry import ToolRegistry


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
    assert model.requests[0][1] == {"role": "user", "content": "Fix it"}


def test_session_keeps_context_across_user_turns() -> None:
    model = FakeModel([AssistantTurn("第一次回答"), AssistantTurn("第二次回答")])
    session = Agent(model, ToolRegistry([]), 3).start_session()

    first = session.ask("第一个问题")
    second = session.ask("继续说明")
    session.close()

    assert first.output == "第一次回答"
    assert second.output == "第二次回答"
    assert model.requests[1] == [
        {"role": "system", "content": model.requests[0][0]["content"]},
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一次回答"},
        {"role": "user", "content": "继续说明"},
    ]


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
        "tool_start",
        "tool_result",
        "model_start",
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
    )

    result = Agent(model, ToolRegistry([EchoTool()]), 2).run("Keep going")

    assert result.reason == "max_iterations"
    assert result.iterations == 2
    assert len(model.requests) == 2


def test_agent_reports_empty_model_response() -> None:
    result = Agent(
        FakeModel([AssistantTurn(None)]), ToolRegistry([]), 2
    ).run("Go")

    assert result.reason == "empty_response"
    assert not result.succeeded


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
