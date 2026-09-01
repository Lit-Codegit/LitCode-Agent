"""Small contextual tools for Session actors.

These tools expose the mailbox model directly.  They intentionally do not
contain a second coordination protocol.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Callable
from pathlib import Path

from litcode_agent.config import CommandPolicy
from litcode_agent.session_runtime import SessionRuntime, SessionRuntimeError
from litcode_agent.session_store import SessionStore
from litcode_agent.tools.base import ToolError, ToolExecutionContext, ToolResult


class SpawnSubagentTool:
    name = "spawn_subagent"
    description = (
        "Create or continue a direct child Session. Foreground calls return the "
        "child's final answer; background calls return its Session id."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "agent": {"type": "string", "enum": ["general", "explore"]},
            "background": {"type": "boolean"},
            "session_id": {"type": "string"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def __init__(self, runtime: SessionRuntime) -> None:
        self.runtime = runtime

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        prompt = _text(arguments, "prompt")
        agent = arguments.get("agent")
        background = arguments.get("background", False)
        session_id = arguments.get("session_id")
        if agent is not None and not isinstance(agent, str):
            raise ToolError("agent must be a string")
        if not isinstance(background, bool):
            raise ToolError("background must be a boolean")
        if session_id is not None and not isinstance(session_id, str):
            raise ToolError("session_id must be a string")
        try:
            result = self.runtime.spawn_subagent(
                context.session_id,
                prompt,
                agent=agent,
                background=background,
                session_id=session_id,
            )
        except SessionRuntimeError as error:
            raise ToolError(str(error)) from error
        payload: dict[str, object] = {
            "invocation_id": result.invocation_id,
            "session_id": result.child_session_id,
            "alias": result.child_alias,
            "background": result.background,
        }
        if result.output is not None:
            payload["output"] = result.output
        return ToolResult(json.dumps(payload, ensure_ascii=False))


class ReadSessionTool:
    name = "read_session"
    description = "Read bounded metadata, activity, queue, and history for a Session."
    input_schema = {
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "query": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 64, "maximum": 8000},
        },
        "required": ["session"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        store: SessionStore,
        workspace: Path,
        policy: CommandPolicy = "allow",
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self.store = store
        self.workspace = workspace.resolve()
        self.policy = policy
        self.confirm = confirm

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.workspace, context)
        _authorize_read(
            self.policy,
            self.confirm,
            "读取另一会话的活动、队列和历史",
            context,
        )
        alias = _text(arguments, "session")
        query = arguments.get("query", "")
        max_chars = arguments.get("max_chars", 2_000)
        if not isinstance(query, str):
            raise ToolError("query must be a string")
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or not 64 <= max_chars <= 8_000:
            raise ToolError("max_chars must be between 64 and 8000")
        try:
            target_id = self.store.session_id_for_reference(self.workspace, alias)
            info = self.store.session_info(target_id)
        except KeyError as error:
            raise ToolError(f"找不到会话（当前工作区）：{alias}") from error
        queue = self.store.queue(target_id, include_finished=False, limit=20)
        excerpt = self.store.search_session_context(
            self.workspace, info.alias, query, max_chars=max_chars
        )
        payload = {
            "session_id": info.id,
            "alias": info.alias,
            "title": info.title,
            "parent_id": info.parent_id,
            "profile": info.profile,
            "status": info.status,
            "paused": info.paused,
            "activity": info.activity,
            "queue": [
                {
                    "id": item.id,
                    "sequence": item.sequence,
                    "source_session_id": item.source_session_id,
                    "content": item.content,
                    "status": item.status,
                }
                for item in queue
            ],
            "history": excerpt,
        }
        return ToolResult(json.dumps(payload, ensure_ascii=False))


class ReadSessionQueueTool:
    name = "read_session_queue"
    description = "Read the complete transparent queue state of a Session."
    input_schema = {
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "include_finished": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["session"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        store: SessionStore,
        workspace: Path,
        policy: CommandPolicy = "allow",
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self.store = store
        self.workspace = workspace.resolve()
        self.policy = policy
        self.confirm = confirm

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.workspace, context)
        _authorize_read(self.policy, self.confirm, "读取另一会话的透明队列", context)
        alias = _text(arguments, "session")
        include_finished = arguments.get("include_finished", False)
        limit = arguments.get("limit", 50)
        if not isinstance(include_finished, bool):
            raise ToolError("include_finished must be a boolean")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ToolError("limit must be between 1 and 100")
        try:
            target_id = self.store.session_id_for_reference(self.workspace, alias)
            rows = self.store.queue(
                target_id, include_finished=include_finished, limit=limit
            )
        except KeyError as error:
            raise ToolError(f"找不到会话（当前工作区）：{alias}") from error
        return ToolResult(
            json.dumps(
                [
                    {
                        "id": item.id,
                        "sequence": item.sequence,
                        "target_session_id": item.target_session_id,
                        "source_session_id": item.source_session_id,
                        "content": item.content,
                        "kind": item.kind,
                        "status": item.status,
                        "created_at": item.created_at,
                    }
                    for item in rows
                ],
                ensure_ascii=False,
            )
        )


class WaitForSessionTool:
    name = "wait_session"
    description = (
        "Block until a background child Session or invocation finishes. "
        "The current turn releases its execution slot while waiting. "
        "Accepts a spawn_subagent invocation id or a Session alias."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "timeout": {"type": "number", "minimum": 1, "maximum": 3600},
        },
        "required": ["session"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        runtime: SessionRuntime,
        policy: CommandPolicy = "allow",
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self.runtime = runtime
        self.policy = policy
        self.confirm = confirm

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.runtime.workspace, context)
        _authorize_read(
            self.policy,
            self.confirm,
            "等待另一会话结束",
            context,
        )
        target = _text(arguments, "session")
        timeout = arguments.get("timeout", 600.0)
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not 1 <= timeout <= 3600
        ):
            raise ToolError("timeout must be between 1 and 3600 seconds")
        try:
            output = self.runtime.wait_for_session(target, float(timeout))
        except SessionRuntimeError as error:
            raise ToolError(str(error)) from error
        if output:
            return ToolResult(
                json.dumps({"session": target, "output": output}, ensure_ascii=False)
            )
        return ToolResult(json.dumps({"session": target, "output": ""}, ensure_ascii=False))


class ControlSessionTool:
    name = "control_session"
    description = "Pause or resume another Session; pausing does not interrupt its current turn."
    input_schema = {
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "action": {"type": "string", "enum": ["pause", "resume"]},
        },
        "required": ["session", "action"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        runtime: SessionRuntime,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self.runtime = runtime
        self.confirm = confirm

    def execute_with_context(
        self, arguments: Mapping[str, object], context: ToolExecutionContext
    ) -> ToolResult:
        _check_workspace(self.runtime.workspace, context)
        alias = _text(arguments, "session")
        action = _text(arguments, "action")
        if action not in {"pause", "resume"}:
            raise ToolError("action must be pause or resume")
        if self.confirm is None:
            raise ToolError("session control requires user confirmation")
        description = f"允许当前 Agent {action} 会话 {alias}？"
        approved = self.runtime.request_confirmation(lambda: self.confirm(description))
        if not approved:
            raise ToolError("session control was not approved")
        try:
            target_id = self.runtime.store.session_id_for_reference(
                self.runtime.workspace, alias
            )
            if action == "pause":
                info = self.runtime.pause(target_id)
            elif action == "resume":
                info = self.runtime.resume(target_id)
        except KeyError as error:
            raise ToolError(f"找不到会话（当前工作区）：{alias}") from error
        except SessionRuntimeError as error:
            raise ToolError(str(error)) from error
        return ToolResult(
            json.dumps(
                {
                    "session_id": info.id,
                    "alias": info.alias,
                    "action": action,
                    "paused": info.paused,
                },
                ensure_ascii=False,
            )
        )


def _text(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} must be a non-empty string")
    return value.strip()


def _authorize_read(
    policy: CommandPolicy,
    confirm: Callable[[str], bool] | None,
    description: str,
    context: ToolExecutionContext,
) -> None:
    if policy == "deny":
        raise ToolError("session read is denied by policy")
    if policy != "confirm":
        return
    if confirm is None:
        approved = False
    elif context.runtime is not None:
        approved = context.runtime.request_confirmation(lambda: confirm(description))
    else:
        approved = confirm(description)
    if not approved:
        raise ToolError("session read was not approved")


def _check_workspace(workspace: Path, context: ToolExecutionContext) -> None:
    if context.workspace.resolve() != workspace.resolve():
        raise ToolError("tool context does not match the configured workspace")
