"""A deterministic whole-loop test using real local tools and no API calls."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from litcode_agent.agent import Agent
from litcode_agent.config import Settings
from litcode_agent.model import AssistantTurn, Message, ToolCall, ToolSchema
from litcode_agent.tools import build_default_registry


class ScriptedCodingModel:
    """Emit a realistic inspect-edit-verify trajectory for integration testing."""

    def __init__(self) -> None:
        self.step = 0

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AssistantTurn:
        expected_tools = {
            "list_files",
            "read_file",
            "search_files",
            "apply_patch",
            "run_command",
            "load_skill",
        }
        assert {
            schema["function"]["name"]  # type: ignore[index]
            for schema in tools
        } == expected_tools
        turns = [
            AssistantTurn(
                None,
                (ToolCall("list", "list_files", '{"path":".","depth":2}'),),
                "tool_calls",
            ),
            AssistantTurn(
                None,
                (ToolCall("read", "read_file", '{"path":"calculator.py"}'),),
                "tool_calls",
            ),
            AssistantTurn(
                None,
                (
                    ToolCall(
                        "edit",
                        "apply_patch",
                        '{"path":"calculator.py","old_text":"return left - right",'
                        '"new_text":"return left + right"}',
                    ),
                ),
                "tool_calls",
            ),
            AssistantTurn(
                None,
                (
                    ToolCall(
                        "verify",
                        "run_command",
                        '{"command":"python -c \\"from calculator import add; '
                        'assert add(2, 3) == 5\\""}',
                    ),
                ),
                "tool_calls",
            ),
            AssistantTurn("Fixed addition and verified add(2, 3) == 5.", (), "stop"),
        ]
        turn = turns[self.step]
        self.step += 1
        return turn


def test_agent_inspects_edits_and_verifies_a_real_workspace(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left - right\n"
    )
    settings = Settings.from_env(
        tmp_path,
        {
            "OPENAI_API_KEY": "unused-in-scripted-test",
            "LITCODE_MODEL": "scripted",
            "LITCODE_COMMAND_POLICY": "deny",
        },
    )

    result = Agent(
        ScriptedCodingModel(),
        build_default_registry(settings),
        max_iterations=8,
    ).run("Fix the add function and verify the result.")

    assert result.succeeded
    assert result.iterations == 5
    assert "return left + right" in (tmp_path / "calculator.py").read_text()
    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    assert len(tool_messages) == 4
    assert '"ok": true' in tool_messages[-1]["content"]
