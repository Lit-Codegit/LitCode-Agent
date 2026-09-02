"""Tool registration, schema export, and recoverable dispatch errors."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from litcode_agent.tools.base import (
    ContextualTool,
    Tool,
    ToolDefinition,
    ToolError,
    ToolExecutionContext,
    ToolResult,
    UserDeclinedError,
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool | ContextualTool]) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, object]]:
        """Return OpenAI-compatible function tool definitions."""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                },
            }
            for tool in self._tools.values()
        ]

    def execute_json(
        self,
        name: str,
        raw_arguments: str,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"unknown tool: {name}")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            return ToolResult.error(f"invalid JSON arguments: {error.msg}")
        if not isinstance(arguments, dict):
            return ToolResult.error("tool arguments must be a JSON object")
        try:
            if context is not None:
                _enforce_profile_policy(name, context)
            if isinstance(tool, ContextualTool):
                if context is None:
                    raise ToolError("tool requires an active session context")
                return tool.execute_with_context(arguments, context)
            assert isinstance(tool, Tool)
            return tool.execute(arguments)
        except UserDeclinedError:
            raise
        except (KeyError, TypeError, ToolError, OSError) as error:
            return ToolResult.error(str(error))


def _enforce_profile_policy(
    tool_name: str, context: ToolExecutionContext
) -> None:
    """Profiles can only narrow the workspace's already granted tools."""

    if context.profile == "explore" and tool_name in {"apply_patch", "run_command"}:
        raise ToolError(f"agent profile explore cannot use {tool_name}")
