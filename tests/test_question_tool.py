"""ask_user 工具的解析、格式化与注册。"""

from __future__ import annotations

from pathlib import Path

import pytest

from litcode_agent.config import Settings
from litcode_agent.tools import build_default_registry
from litcode_agent.tools.base import ToolError, ToolExecutionContext
from litcode_agent.tools.question import AskUserTool, QuestionSpec


def context(session_id: str = "s1") -> ToolExecutionContext:
    return ToolExecutionContext(session_id, Path("."))


def _example_questions() -> dict[str, object]:
    return {
        "questions": [
            {
                "header": "方向",
                "question": "继续还是回退？",
                "options": [
                    {"label": "继续", "description": "按计划继续"},
                    {"label": "回退", "description": "撤销前一步"},
                ],
            }
        ]
    }


def test_ask_user_returns_formatted_answers() -> None:
    answered: list[list[str]] = []

    def ask(session_id: str, questions: list[QuestionSpec]) -> list[list[str]]:
        assert session_id == "s1"
        assert questions[0].header == "方向"
        answered.append([questions[0].options[0][0]])
        return list(answered)

    result = AskUserTool(ask).execute_with_context(
        {"questions": _example_questions()["questions"]}, context()
    )

    assert not result.is_error
    assert (
        'User has answered your questions: "继续还是回退？"="继续". '
        "You can now continue with the user's answers in mind."
    ) == result.content


def test_ask_user_without_ui_raises_recoverable_error(tmp_path: Path) -> None:
    import json

    from litcode_agent.tools.registry import ToolRegistry

    registry = ToolRegistry([AskUserTool()])
    result = registry.execute_json(
        "ask_user", json.dumps(_example_questions()), context()
    )

    assert result.is_error
    assert "交互终端" in result.content


def test_ask_user_rejects_malformed_input() -> None:
    tool = AskUserTool(lambda sid, qs: [])
    with pytest.raises(ToolError):
        tool.execute_with_context({"questions": []}, context())
    with pytest.raises(ToolError):
        tool.execute_with_context({"questions": [{"header": "h"}]}, context())
    import litcode_agent.tools.question as module

    with pytest.raises(ToolError):
        module._parse_questions(
            [
                {
                    "header": "h",
                    "question": "q",
                    "options": [{"label": "a"}],
                }
            ]
        )
    with pytest.raises(ToolError):
        module._parse_questions(
            [
                {
                    "header": "h",
                    "question": "q",
                    "options": [{"label": "a", "description": "b"}],
                    "multiple": "yes",
                }
            ]
        )


def test_ask_user_registered_in_default_registry(tmp_path: Path) -> None:
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "example-model"},
    )
    names = {
        schema["function"]["name"]  # type: ignore[index]
        for schema in build_default_registry(settings).schemas()
    }

    assert "ask_user" in names


def test_ask_user_caps_header_length() -> None:
    from litcode_agent.tools.question import _parse_questions

    header = "x" * 80
    parsed = _parse_questions(
        [
            {
                "header": header,
                "question": "q",
                "options": [{"label": "a", "description": "b"}],
            }
        ]
    )

    assert len(parsed[0].header) <= 30
    assert parsed[0].header.endswith("…")
