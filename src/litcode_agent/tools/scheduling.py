"""Contextual tools for creating and managing scheduled Agent turns."""

from __future__ import annotations

import json
from collections.abc import Mapping

from litcode_agent.scheduler import Scheduler, ScheduleError, normalize_schedule
from litcode_agent.session_store import SessionStore
from litcode_agent.tools.base import ToolError, ToolExecutionContext, ToolResult


class CreateScheduledTaskTool:
    name = "create_scheduled_task"
    description = (
        "Create a durable scheduled Agent task only when the user explicitly asks. "
        "Normalize the user's natural-language date using the current local date. "
        "The task starts a full Agent turn in a dedicated child Session."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Complete instruction the future Agent should perform.",
            },
            "kind": {"type": "string", "enum": ["once", "daily", "weekly", "monthly"]},
            "timezone": {
                "type": "string",
                "description": "IANA timezone, for example Asia/Shanghai.",
            },
            "run_at": {
                "type": "string",
                "description": "ISO 8601 local or offset datetime; required for once.",
            },
            "time": {
                "type": "string",
                "description": "24-hour HH:MM wall time; required for recurring tasks.",
            },
            "weekdays": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 7},
                "description": "ISO weekdays Monday=1..Sunday=7; required for weekly.",
            },
            "day_of_month": {
                "type": "integer",
                "minimum": 1,
                "maximum": 31,
                "description": "Required for monthly; months without this day are skipped.",
            },
        },
        "required": ["prompt", "kind", "timezone"],
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore, scheduler: Scheduler | None = None) -> None:
        self.store = store
        self.scheduler = scheduler

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolError("prompt must be a non-empty string")
        try:
            rule, zone_name, first = normalize_schedule(arguments)
        except ScheduleError as error:
            raise ToolError(str(error)) from error
        child_id = self.store.create_child(
            context.session_id,
            title=f"定时任务：{prompt.strip()[:40]}",
            profile="general",
        )
        task = self.store.create_scheduled_task(
            context.session_id,
            child_id,
            prompt,
            rule,
            zone_name,
            first,
        )
        if self.scheduler is not None:
            self.scheduler.notify()
        return ToolResult(json.dumps(_task_payload(task), ensure_ascii=False))


class ListScheduledTasksTool:
    name = "list_scheduled_tasks"
    description = "List durable scheduled Agent tasks in the current workspace."
    input_schema = {
        "type": "object",
        "properties": {"include_inactive": {"type": "boolean"}},
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        include = arguments.get("include_inactive", False)
        if not isinstance(include, bool):
            raise ToolError("include_inactive must be a boolean")
        tasks = self.store.scheduled_tasks(
            context.workspace, include_inactive=include
        )
        return ToolResult(
            json.dumps([_task_payload(task) for task in tasks], ensure_ascii=False)
        )


class CancelScheduledTaskTool:
    name = "cancel_scheduled_task"
    description = "Cancel an active scheduled Agent task in the current workspace."
    input_schema = {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore, scheduler: Scheduler | None = None) -> None:
        self.store = store
        self.scheduler = scheduler

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ToolError("task_id must be a non-empty string")
        try:
            task = self.store.cancel_scheduled_task(context.workspace, task_id)
        except (KeyError, ValueError) as error:
            raise ToolError(str(error)) from error
        if self.scheduler is not None:
            self.scheduler.notify()
        return ToolResult(json.dumps(_task_payload(task), ensure_ascii=False))


def _task_payload(task) -> dict[str, object]:
    return {
        "task_id": task.id,
        "session_id": task.target_session_id,
        "prompt": task.prompt,
        "schedule": dict(task.schedule),
        "timezone": task.timezone,
        "next_run_at": task.next_run_at,
        "status": task.status,
    }
