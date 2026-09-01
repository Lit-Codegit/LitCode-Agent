"""Local tools exposed to the language model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from litcode_agent.config import Settings
from litcode_agent.tools.base import ToolResult
from litcode_agent.tools.command import ConfirmCommand, RunCommandTool
from litcode_agent.tools.files import (
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
)
from litcode_agent.tools.question import AskUserCallback, AskUserTool, QuestionSpec
from litcode_agent.tools.registry import ToolRegistry
from litcode_agent.tools.workspace import Workspace
from litcode_agent.read_scope import ReadScope
from litcode_agent.skills import SkillCatalog
from litcode_agent.tools.skills import LoadSkillTool
from litcode_agent.mutation_locks import WorkspaceMutationLocks

if TYPE_CHECKING:
    from litcode_agent.scheduler import Scheduler
    from litcode_agent.session_store import SessionStore
    from litcode_agent.session_runtime import SessionRuntime


def build_default_registry(
    settings: Settings,
    confirm: ConfirmCommand | None = None,
    skills: SkillCatalog | None = None,
    store: SessionStore | None = None,
    confirm_session_message: Callable[[str], bool] | None = None,
    confirm_session_read: Callable[[str], bool] | None = None,
    runtime: SessionRuntime | None = None,
    scheduler: Scheduler | None = None,
    confirm_session_control: Callable[[str], bool] | None = None,
    ask_user: AskUserCallback | None = None,
) -> ToolRegistry:
    """Construct the complete MVP tool set from validated settings."""

    workspace = Workspace(settings.workspace)
    read_scope = ReadScope(workspace, settings.read_roots)
    skill_catalog = skills or SkillCatalog.discover(settings.workspace)
    execution_lock = WorkspaceMutationLocks.for_workspace(workspace.root)
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
            runtime,
        ),
        LoadSkillTool(skill_catalog, settings.max_output_chars),
        AskUserTool(ask_user),
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
                ListSessionsTool(
                    store,
                    settings.workspace,
                    settings.session_read_policy,
                    confirm_session_read,
                ),
                ReadSessionContextTool(
                    store,
                    settings.workspace,
                    settings.session_read_policy,
                    confirm_session_read,
                ),
                SendSessionMessageTool(
                    store,
                    settings.workspace,
                    settings.session_message_policy,
                    confirm_session_message,
                    runtime,
                ),
                ReadSessionInboxTool(
                    store,
                    settings.workspace,
                    settings.session_read_policy,
                    confirm_session_read,
                ),
            ]
        )
        if runtime is not None:
            from litcode_agent.tools.subagents import (
                ControlSessionTool,
                ReadSessionTool,
                ReadSessionQueueTool,
                SpawnSubagentTool,
                WaitForSessionTool,
            )

            tools.extend(
                [
                    SpawnSubagentTool(runtime),
                    WaitForSessionTool(
                        runtime,
                        settings.session_read_policy,
                        confirm_session_read,
                    ),
                    ReadSessionTool(
                        store,
                        settings.workspace,
                        settings.session_read_policy,
                        confirm_session_read,
                    ),
                    ReadSessionQueueTool(
                        store,
                        settings.workspace,
                        settings.session_read_policy,
                        confirm_session_read,
                    ),
                    ControlSessionTool(runtime, confirm_session_control),
                ]
            )
            from litcode_agent.tools.scheduling import (
                CancelScheduledTaskTool,
                CreateScheduledTaskTool,
                ListScheduledTasksTool,
            )

            tools.extend(
                [
                    CreateScheduledTaskTool(store, scheduler),
                    ListScheduledTasksTool(store),
                    CancelScheduledTaskTool(store, scheduler),
                ]
            )
    return ToolRegistry(tools)

__all__ = [
    "ApplyPatchTool",
    "AskUserTool",
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
