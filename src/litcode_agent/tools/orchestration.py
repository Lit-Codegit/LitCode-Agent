"""Structured orchestration tools with trusted runtime identities."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from litcode_agent.config import CommandPolicy
from litcode_agent.orchestration import (
    OrchestrationError,
    OrchestrationService,
    OrchestrationTask,
)
from litcode_agent.tools.base import ToolError, ToolExecutionContext, ToolResult

OrchestrationChange = Callable[[OrchestrationTask], None]
ConfirmWake = Callable[[str], bool]


class DelegateSessionTool:
    name = "delegate_session"
    description = (
        "Queue one bounded task for another session in an approved orchestration run."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "session": {"type": "string"},
            "role": {"type": "string", "enum": ["implementer", "reviewer"]},
            "objective": {"type": "string"},
            "acceptance": {"type": "array", "items": {"type": "string"}},
            "allowed_paths": {"type": "array", "items": {"type": "string"}},
            "write_policy": {
                "type": "string",
                "enum": ["none", "workspace-write"],
            },
        },
        "required": [
            "run_id",
            "session",
            "role",
            "objective",
            "acceptance",
            "write_policy",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        service: OrchestrationService,
        on_change: OrchestrationChange | None = None,
        policy: CommandPolicy = "confirm",
        confirm: ConfirmWake | None = None,
    ) -> None:
        self.service = service
        self.on_change = on_change
        self.policy = policy
        self.confirm = confirm

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        target = _text(arguments, "session")
        objective = _text(arguments, "objective")
        if self.policy == "deny":
            raise ToolError("automatic session wake is denied by policy")
        if self.policy == "confirm" and (
            self.confirm is None
            or not self.confirm(f"允许编排唤醒会话 {target}：\n{objective}")
        ):
            raise ToolError("automatic session wake was not approved")
        try:
            task = self.service.delegate(
                _text(arguments, "run_id"),
                context.session_id,
                target,
                role=_text(arguments, "role"),  # type: ignore[arg-type]
                objective=objective,
                acceptance=_texts(arguments, "acceptance"),
                allowed_paths=_texts(arguments, "allowed_paths", default=()),
                write_policy=_text(arguments, "write_policy"),  # type: ignore[arg-type]
            )
        except (OrchestrationError, KeyError) as error:
            raise ToolError(str(error)) from error
        if self.on_change is not None:
            self.on_change(task)
        return ToolResult(
            json.dumps(
                {
                    "task_id": task.id,
                    "run_id": task.run_id,
                    "status": task.status,
                    "target_session_id": task.target_session_id,
                    "note": "queued 不等于目标已经运行",
                },
                ensure_ascii=False,
            )
        )


class ReportTaskTool:
    name = "report_task"
    description = "Report the structured terminal result of the active assigned task."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["completed", "blocked", "failed"],
            },
            "summary": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "changed_files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["task_id", "status", "summary"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        service: OrchestrationService,
        on_change: OrchestrationChange | None = None,
    ) -> None:
        self.service = service
        self.on_change = on_change

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        try:
            task = self.service.report_task(
                _text(arguments, "task_id"),
                context.session_id,
                status=_text(arguments, "status"),  # type: ignore[arg-type]
                summary=_text(arguments, "summary"),
                evidence=_texts(arguments, "evidence", default=()),
                changed_files=_texts(arguments, "changed_files", default=()),
            )
        except (OrchestrationError, KeyError) as error:
            raise ToolError(str(error)) from error
        if self.on_change is not None:
            self.on_change(task)
        return ToolResult(
            json.dumps(
                {"task_id": task.id, "run_id": task.run_id, "status": task.status},
                ensure_ascii=False,
            )
        )


class ListOrchestrationTool:
    name = "list_orchestration"
    description = "Read the bounded causal ledger for one orchestration run."
    input_schema = {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
        "additionalProperties": False,
    }

    def __init__(self, service: OrchestrationService) -> None:
        self.service = service

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        try:
            events = self.service.ledger(_text(arguments, "run_id"))
        except (OrchestrationError, KeyError) as error:
            raise ToolError(str(error)) from error
        return ToolResult(
            json.dumps(
                [
                    {
                        "id": event.id,
                        "task_id": event.task_id,
                        "kind": event.kind,
                        "source_session_id": event.source_session_id,
                        "target_session_id": event.target_session_id,
                        "summary": event.summary,
                        "created_at": event.created_at,
                    }
                    for event in events[-50:]
                ],
                ensure_ascii=False,
            )
        )


class FinishOrchestrationTool:
    name = "finish_orchestration"
    description = "Finish an orchestration after all queued/running tasks have ended."
    input_schema = {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["completed", "failed", "cancelled"],
            },
            "summary": {"type": "string"},
        },
        "required": ["run_id", "status", "summary"],
        "additionalProperties": False,
    }

    def __init__(self, service: OrchestrationService) -> None:
        self.service = service

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        try:
            run = self.service.finish_run(
                _text(arguments, "run_id"),
                context.session_id,
                status=_text(arguments, "status"),  # type: ignore[arg-type]
                summary=_text(arguments, "summary"),
            )
        except (OrchestrationError, KeyError) as error:
            raise ToolError(str(error)) from error
        return ToolResult(
            json.dumps(
                {"run_id": run.id, "status": run.status}, ensure_ascii=False
            )
        )


def _text(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} must be a non-empty string")
    return value.strip()


def _texts(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    value = arguments.get(name)
    if value is None and default is not None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ToolError(f"{name} must be an array of strings")
    return tuple(value)
