"""Model boundary and OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from openai import OpenAI, OpenAIError

from litcode_agent.config import Settings

Message = dict[str, Any]
ToolSchema = dict[str, object]


class ModelError(RuntimeError):
    """A model request or response could not be used by the agent."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_message_item(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None

    def as_message(self) -> Message:
        message: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.as_message_item() for tool_call in self.tool_calls
            ]
        return message


class Model(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> AssistantTurn: ...


class OpenAIChatModel:
    """Normalize SDK objects so the agent loop has no OpenAI dependency."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.model = settings.model
        self.client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=0,
        )

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> AssistantTurn:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                tools=list(tools),
                tool_choice="auto",
            )
        except OpenAIError as error:
            raise ModelError(f"model request failed: {error}") from error
        if not completion.choices:
            raise ModelError("model response contained no choices")

        choice = completion.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or ():
            if call.type != "function":
                raise ModelError(f"unsupported model tool call type: {call.type}")
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )
        return AssistantTurn(
            content=message.content,
            tool_calls=tuple(tool_calls),
            finish_reason=getattr(choice, "finish_reason", None),
        )
