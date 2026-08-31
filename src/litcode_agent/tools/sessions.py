"""Bounded same-workspace session inspection and explicit inbox delivery."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Mapping

from litcode_agent.config import CommandPolicy
from litcode_agent.session_store import SessionStore
from litcode_agent.tools.base import ToolError, ToolExecutionContext, ToolResult

ConfirmSessionMessage = Callable[[str], bool]


class ListSessionsTool:
    name = "list_sessions"
    description = "List bounded metadata for sessions in the current workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore, workspace: Path) -> None:
        self.store = store
        self.workspace = workspace.resolve()

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.workspace, context)
        query = arguments.get("query", "")
        limit = arguments.get("limit", 20)
        if not isinstance(query, str):
            raise ToolError("query must be a string")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ToolError("limit must be between 1 and 50")
        needle = query.casefold()
        rows = [
            item
            for item in self.store.list_sessions(self.workspace, limit=50)
            if not needle
            or needle in item.alias.casefold()
            or needle in item.title.casefold()
        ][:limit]
        return ToolResult(
            json.dumps(
                [
                    {
                        "alias": item.alias,
                        "title": item.title,
                        "model": item.model,
                        "updated_at": item.updated_at,
                    }
                    for item in rows
                ],
                ensure_ascii=False,
            )
        )


class ReadSessionContextTool:
    name = "read_session_context"
    description = (
        "Search one same-workspace session and return only bounded matching excerpts."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "query": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 64, "maximum": 8000},
        },
        "required": ["session", "query"],
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore, workspace: Path) -> None:
        self.store = store
        self.workspace = workspace.resolve()

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.workspace, context)
        alias = _string(arguments, "session")
        query = _string(arguments, "query")
        max_chars = arguments.get("max_chars", 2000)
        if (
            not isinstance(max_chars, int)
            or isinstance(max_chars, bool)
            or not 64 <= max_chars <= 8000
        ):
            raise ToolError("max_chars must be between 64 and 8000")
        try:
            excerpt = self.store.search_session_context(
                self.workspace, alias, query, max_chars=max_chars
            )
        except KeyError as error:
            raise ToolError(f"找不到会话（当前工作区）：{alias}") from error
        return ToolResult(excerpt)


class SendSessionMessageTool:
    name = "send_session_message"
    description = (
        "Deliver an instruction to another same-workspace session inbox. "
        "This never wakes or runs the target model automatically."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "instruction": {"type": "string"},
        },
        "required": ["session", "instruction"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        store: SessionStore,
        workspace: Path,
        policy: CommandPolicy,
        confirm: ConfirmSessionMessage | None,
    ) -> None:
        self.store = store
        self.workspace = workspace.resolve()
        self.policy = policy
        self.confirm = confirm

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.workspace, context)
        alias = _string(arguments, "session")
        instruction = _string(arguments, "instruction")
        description = f"向会话 {alias} 投递指示：\n{instruction}"
        if self.policy == "deny":
            raise ToolError("session messaging is denied by policy")
        if self.policy == "confirm" and (
            self.confirm is None or not self.confirm(description)
        ):
            raise ToolError("session message was not approved")
        try:
            message = self.store.send_to_session(
                self.workspace, context.session_id, alias, instruction
            )
        except KeyError as error:
            raise ToolError(f"找不到会话（当前工作区）：{alias}") from error
        return ToolResult(
            f"已投递到 {alias}；message_id={message.id}；目标模型未自动启动。"
        )


class ReadSessionInboxTool:
    name = "read_session_inbox"
    description = "Read unread instructions delivered to the active session inbox."
    input_schema = {
        "type": "object",
        "properties": {"mark_read": {"type": "boolean"}},
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore, workspace: Path) -> None:
        self.store = store
        self.workspace = workspace.resolve()

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.workspace, context)
        mark_read = arguments.get("mark_read", True)
        if not isinstance(mark_read, bool):
            raise ToolError("mark_read must be a boolean")
        messages = self.store.inbox(context.session_id)
        payload = []
        for message in messages:
            source = self.store.session_info(message.source_session_id)
            payload.append(
                {
                    "message_id": message.id,
                    "source": source.alias,
                    "content": message.content,
                    "created_at": message.created_at,
                }
            )
            if mark_read:
                self.store.mark_inbox_read(context.session_id, message.id)
        return ToolResult(json.dumps(payload, ensure_ascii=False))


def _string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} must be a non-empty string")
    return value.strip()


def _check_workspace(workspace: Path, context: ToolExecutionContext) -> None:
    if context.workspace.resolve() != workspace:
        raise ToolError("tool context does not match the configured workspace")
