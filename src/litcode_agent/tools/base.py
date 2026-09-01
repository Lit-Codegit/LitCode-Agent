"""Shared tool interfaces and results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable


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


class ToolDefinition(Protocol):
    """Metadata shared by ordinary and session-context tools."""

    name: str
    description: str
    input_schema: Mapping[str, object]


@runtime_checkable
class Tool(ToolDefinition, Protocol):
    """A local tool whose caller identity is irrelevant."""

    def execute(self, arguments: Mapping[str, object]) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Trusted invocation identity supplied by the Agent runtime."""

    session_id: str
    workspace: Path
    terminal_id: str | None = None
    pane_slot: int | None = None
    mounted_sessions: tuple[tuple[str, int], ...] = ()
    orchestration_role: str | None = None
    orchestration_write_policy: str | None = None
    orchestration_allowed_paths: tuple[str, ...] = ()


@runtime_checkable
class ContextualTool(ToolDefinition, Protocol):
    """A tool that requires an unforgeable active-session context."""

    def execute_with_context(
        self,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> ToolResult: ...
