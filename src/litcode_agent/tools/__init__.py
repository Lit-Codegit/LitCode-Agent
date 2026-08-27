"""Local tools exposed to the language model."""

from __future__ import annotations

from litcode_agent.config import Settings
from litcode_agent.tools.base import ToolResult
from litcode_agent.tools.command import ConfirmCommand, RunCommandTool
from litcode_agent.tools.files import (
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
)
from litcode_agent.tools.registry import ToolRegistry
from litcode_agent.tools.workspace import Workspace


def build_default_registry(
    settings: Settings, confirm: ConfirmCommand | None = None
) -> ToolRegistry:
    """Construct the complete MVP tool set from validated settings."""

    workspace = Workspace(settings.workspace)
    return ToolRegistry(
        [
            ListFilesTool(workspace, settings.max_output_chars),
            ReadFileTool(workspace, settings.max_output_chars),
            SearchFilesTool(
                workspace,
                settings.max_output_chars,
                settings.command_timeout_seconds,
            ),
            ApplyPatchTool(workspace),
            RunCommandTool(
                workspace,
                settings.command_timeout_seconds,
                settings.max_output_chars,
                settings.command_policy,
                confirm,
            ),
        ]
    )

__all__ = [
    "ApplyPatchTool",
    "ListFilesTool",
    "ReadFileTool",
    "RunCommandTool",
    "SearchFilesTool",
    "ToolRegistry",
    "ToolResult",
    "Workspace",
    "build_default_registry",
]
