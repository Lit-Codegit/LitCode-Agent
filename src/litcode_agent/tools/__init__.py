"""Local tools exposed to the language model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
import threading

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
from litcode_agent.read_scope import ReadScope
from litcode_agent.skills import SkillCatalog
from litcode_agent.tools.skills import LoadSkillTool

if TYPE_CHECKING:
    from litcode_agent.session_store import SessionStore


def build_default_registry(
    settings: Settings,
    confirm: ConfirmCommand | None = None,
    skills: SkillCatalog | None = None,
    store: SessionStore | None = None,
    confirm_session_message: Callable[[str], bool] | None = None,
) -> ToolRegistry:
    """Construct the complete MVP tool set from validated settings."""

    workspace = Workspace(settings.workspace)
    read_scope = ReadScope(workspace, settings.read_roots)
    skill_catalog = skills or SkillCatalog.discover(settings.workspace)
    execution_lock = threading.RLock()
    tools = [
        ListFilesTool(read_scope, settings.max_output_chars),
        ReadFileTool(read_scope, settings.max_output_chars),
        SearchFilesTool(
            read_scope,
            settings.max_output_chars,
            settings.command_timeout_seconds,
        ),
        ApplyPatchTool(workspace, execution_lock),
        RunCommandTool(
            workspace,
            settings.command_timeout_seconds,
            settings.max_output_chars,
            settings.command_policy,
            confirm,
            execution_lock,
        ),
        LoadSkillTool(skill_catalog, settings.max_output_chars),
    ]
    if store is not None:
        from litcode_agent.tools.sessions import (
            ListSessionsTool,
            ReadSessionContextTool,
            ReadSessionInboxTool,
            SendSessionMessageTool,
        )

        tools.extend(
            [
                ListSessionsTool(store, settings.workspace),
                ReadSessionContextTool(store, settings.workspace),
                SendSessionMessageTool(
                    store,
                    settings.workspace,
                    settings.session_message_policy,
                    confirm_session_message,
                ),
                ReadSessionInboxTool(store, settings.workspace),
            ]
        )
    return ToolRegistry(tools)

__all__ = [
    "ApplyPatchTool",
    "ListFilesTool",
    "LoadSkillTool",
    "ReadFileTool",
    "RunCommandTool",
    "SearchFilesTool",
    "ToolRegistry",
    "ToolResult",
    "Workspace",
    "build_default_registry",
]
