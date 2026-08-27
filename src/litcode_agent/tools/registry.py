"""Tool registration, schema export, and recoverable dispatch errors."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from litcode_agent.tools.base import Tool, ToolError, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
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

    def execute_json(self, name: str, raw_arguments: str) -> ToolResult:
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
            return tool.execute(arguments)
        except (ToolError, OSError) as error:
            return ToolResult.error(str(error))
