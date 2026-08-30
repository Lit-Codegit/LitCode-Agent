"""Shared tool interfaces and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


class ToolError(ValueError):
    """An expected error that the model may correct on its next turn."""


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    before_content: str | None
    after_content: str
    before_exists: bool


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Text returned to the model after a local tool invocation."""

    content: str
    is_error: bool = False
    file_change: FileChange | None = None

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(content=message, is_error=True)


class Tool(Protocol):
    """The small interface implemented by every local tool."""

    name: str
    description: str
    input_schema: Mapping[str, object]

    def execute(self, arguments: Mapping[str, object]) -> ToolResult: ...
