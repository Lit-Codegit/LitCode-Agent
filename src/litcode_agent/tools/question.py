"""Ask the user a structured question during an interactive session.

接口形状借鉴 OpenCode 的 Question 工具（`packages/opencode/src/tool/question.ts`
与 `packages/tui/src/routes/session/question.tsx`）：模型传入带选项的问题列表，
UI 以全屏选择器收集答案，工具结果以 `"q"="answer"` 摘要返回模型。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from litcode_agent.tools.base import (
    ContextualTool,
    ToolError,
    ToolResult,
    ToolExecutionContext,
)

MAX_QUESTIONS = 4
MAX_OPTIONS = 9
MAX_HEADER_CHARS = 30


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """One user question with bounded options for the picker UI."""

    header: str
    question: str
    options: tuple[tuple[str, str], ...]
    multiple: bool = False
    custom: bool = True


AskUserCallback = Callable[[str, list[QuestionSpec]], list[list[str]]]

_DESCRIPTION = (
    "Ask the user a question during execution. Use this to gather preferences, "
    "clarify ambiguous instructions, or get an explicit decision on "
    "implementation choices before committing to a direction. "
    "Answers are returned as arrays of selected labels. Set `multiple: true` "
    "only when more than one answer is valid. When `custom` is enabled "
    "(default), the UI adds a type-your-own-answer field, so do not add an "
    "\"Other\" catch-all option. If you recommend one option, list it first and "
    "append \"(Recommended)\" to its label. Ask only when the answer actually "
    "shapes the work; otherwise continue with the best default."
)


class AskUserTool(ContextualTool):
    name = "ask_user"
    description = _DESCRIPTION
    input_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "string",
                            "description": "Very short label shown above the question",
                        },
                        "question": {"type": "string", "description": "Complete question"},
                        "options": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_OPTIONS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["label", "description"],
                                "additionalProperties": False,
                            },
                        },
                        "multiple": {"type": "boolean"},
                        "custom": {"type": "boolean"},
                    },
                    "required": ["header", "question", "options"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }

    def __init__(self, ask: AskUserCallback | None = None) -> None:
        self.ask = ask

    def execute_with_context(
        self,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> ToolResult:
        questions = _parse_questions(arguments.get("questions"))
        if self.ask is None:
            raise ToolError(
                "ask_user 只在交互终端会话中可用（非交互 run 模式无法提问）。"
            )
        runtime = context.runtime
        wait = lambda: self.ask(context.session_id, questions)
        if runtime is not None:
            answers = runtime.request_user_answer(wait)
            if answers is None:
                raise ToolError("用户取消了提问。")
        else:
            answers = wait()
        return ToolResult(_format_answers(questions, answers))


def _parse_questions(raw: object) -> list[QuestionSpec]:
    if not isinstance(raw, list) or not raw:
        raise ToolError("questions must be a non-empty array")
    if len(raw) > MAX_QUESTIONS:
        raise ToolError(f"questions must not exceed {MAX_QUESTIONS} entries")
    parsed: list[QuestionSpec] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ToolError(f"questions[{index}] must be an object")
        header = _text(entry, "header", index)
        question = _text(entry, "question", index)
        if len(header) > MAX_HEADER_CHARS:
            header = f"{header[:MAX_HEADER_CHARS - 1]}…"
        options = _parse_options(entry.get("options"), index)
        multiple = _boolean(entry.get("multiple", False), "multiple", index)
        custom = _boolean(entry.get("custom", True), "custom", index)
        parsed.append(QuestionSpec(header, question, options, multiple, custom))
    return parsed


def _parse_options(raw: object, question_index: int) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise ToolError(
            f"questions[{question_index}].options must be a non-empty array"
        )
    if len(raw) > MAX_OPTIONS:
        raise ToolError(f"questions[{question_index}] has too many options")
    options: list[tuple[str, str]] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ToolError(
                f"questions[{question_index}].options[{index}] must be an object"
            )
        label = _text(entry, "label", question_index, prefix=f"options[{index}]")
        description = _text(
            entry, "description", question_index, prefix=f"options[{index}]"
        )
        options.append((label, description))
    return tuple(options)


def _text(
    entry: Mapping[str, object], name: str, question_index: int, *, prefix: str = ""
) -> str:
    value = entry.get(name)
    path = prefix or name
    if not isinstance(value, str) or not value.strip():
        raise ToolError(
            f"questions[{question_index}].{path} must be a non-empty string"
        )
    return " ".join(value.split())


def _boolean(value: object, name: str, question_index: int) -> bool:
    if not isinstance(value, bool):
        raise ToolError(f"questions[{question_index}].{name} must be a boolean")
    return value


def _format_answers(
    questions: list[QuestionSpec], answers: list[list[str]]
) -> str:
    rendered = []
    for index, question in enumerate(questions):
        chosen = answers[index] if index < len(answers) else []
        if isinstance(chosen, str):
            chosen = [chosen]
        selected = [answer for answer in chosen if answer]
        label = ", ".join(selected) if selected else "Unanswered"
        rendered.append(f'"{question.question}"="{label}"')
    joined = ", ".join(rendered)
    return (
        f"User has answered your questions: {joined}. "
        "You can now continue with the user's answers in mind."
    )
