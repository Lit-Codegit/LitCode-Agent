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
