"""Model boundary and OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

from collections.abc import Iterator
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


@dataclass(frozen=True, slots=True)
class ModelDelta:
    """一个与厂商 SDK 无关的流式响应片段。"""

    content: str = ""
    tool_index: int | None = None
    tool_call_id: str = ""
    tool_name: str = ""
    tool_arguments: str = ""
    finish_reason: str | None = None


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

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> Iterator[ModelDelta]:
        """把 OpenAI-compatible chunks 归一化为可累计的 delta。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                tools=list(tools),
                tool_choice="auto",
                stream=True,
            )
            try:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    content = getattr(delta, "content", None) or ""
                    finish_reason = getattr(choice, "finish_reason", None)
                    tool_chunks = getattr(delta, "tool_calls", None) or ()
                    if not tool_chunks:
                        if content or finish_reason:
                            yield ModelDelta(
                                content=content,
                                finish_reason=finish_reason,
                            )
                        continue
                    for tool_chunk in tool_chunks:
                        function = getattr(tool_chunk, "function", None)
                        yield ModelDelta(
                            content=content,
                            tool_index=tool_chunk.index,
                            tool_call_id=getattr(tool_chunk, "id", None) or "",
                            tool_name=(
                                getattr(function, "name", None) or ""
                                if function is not None
                                else ""
                            ),
                            tool_arguments=(
                                getattr(function, "arguments", None) or ""
                                if function is not None
                                else ""
                            ),
                            finish_reason=finish_reason,
                        )
                        content = ""
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except OpenAIError as error:
            raise ModelError(f"model stream failed: {error}") from error

    def list_models(self) -> tuple[str, ...]:
        """查询当前 API 端点公开的模型 ID。"""

        try:
            response = self.client.models.list()
        except OpenAIError as error:
            raise ModelError(f"model query failed: {error}") from error
        model_ids = {
            model.id
            for model in response.data
            if isinstance(getattr(model, "id", None), str) and model.id
        }
        return tuple(sorted(model_ids))

    def select_model(self, model: str) -> None:
        """切换当前会话后续请求使用的具体模型 ID。"""

        if not model.strip():
            raise ValueError("model must not be empty")
        self.model = model.strip()
