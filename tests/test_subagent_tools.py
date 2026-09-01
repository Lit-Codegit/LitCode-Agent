from __future__ import annotations

import json
from pathlib import Path

import pytest

from litcode_agent.session_runtime import SessionRuntime
from litcode_agent.session_store import SessionStore
from litcode_agent.tools.base import ToolExecutionContext
from litcode_agent.tools.subagents import (
    ControlSessionTool,
    ReadSessionQueueTool,
    ReadSessionTool,
    SpawnSubagentTool,
    WaitForSessionTool,
)


def test_session_read_tools_expose_metadata_queue_and_history(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    target = store.create(tmp_path, "model", [], profile="explore")
    store.save_messages(
        target,
        [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "调查结论"},
        ],
        title="只读调查",
    )
    store.enqueue_message(target, "请检查日志", source_session_id=source)
    context = ToolExecutionContext(source, tmp_path.resolve())

    read = ReadSessionTool(store, tmp_path).execute_with_context(
        {"session": target, "query": "调查"}, context
    )
    payload = json.loads(read.content)
    assert payload["session_id"] == target
    assert payload["profile"] == "explore"
    assert "调查结论" in payload["history"]
    assert payload["queue"][0]["content"] == "请检查日志"

    queue = ReadSessionQueueTool(store, tmp_path).execute_with_context(
        {"session": store.session_info(target).alias}, context
    )
    assert json.loads(queue.content)[0]["target_session_id"] == target


def test_session_control_requires_confirmation_and_changes_pause_gate(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    target = store.create(tmp_path, "model", [])
    runtime = SessionRuntime(store, tmp_path, claim_workspace=False)
    context = ToolExecutionContext(source, tmp_path.resolve())

    with pytest.raises(ValueError, match="confirmation"):
        ControlSessionTool(runtime).execute_with_context(
            {"session": target, "action": "pause"}, context
        )

    allowed = ControlSessionTool(runtime, lambda _: True).execute_with_context(
        {"session": target, "action": "pause"}, context
    )
    assert not allowed.is_error
    assert store.session_info(target).paused
    runtime.close()


def test_spawn_tool_rejects_calls_without_an_owned_parent_turn(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    runtime = SessionRuntime(store, tmp_path, claim_workspace=False)
    with pytest.raises(ValueError, match="active parent turn"):
        SpawnSubagentTool(runtime).execute_with_context(
            {"prompt": "调查"}, ToolExecutionContext(source, tmp_path.resolve())
        )
    runtime.close()


def test_wait_session_tool_authorizes_reads_and_reports_wait_errors(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    target = store.create(tmp_path, "model", [])
    runtime = SessionRuntime(store, tmp_path, claim_workspace=False)
    context = ToolExecutionContext(source, tmp_path.resolve())

    with pytest.raises(ValueError, match="not approved"):
        WaitForSessionTool(runtime, "confirm").execute_with_context(
            {"session": store.session_info(target).alias, "timeout": 1}, context
        )

    with pytest.raises(ValueError, match="active parent turn"):
        WaitForSessionTool(runtime, "allow").execute_with_context(
            {"session": target}, context
        )
    with pytest.raises(ValueError, match="timeout must"):
        WaitForSessionTool(runtime, "allow").execute_with_context(
            {"session": target, "timeout": True}, context
        )
    runtime.close()


def test_session_tools_return_recoverable_errors_for_bad_context_and_arguments(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    runtime = SessionRuntime(store, tmp_path, claim_workspace=False)
    tool = ReadSessionQueueTool(store, tmp_path)

    with pytest.raises(ValueError, match="workspace"):
        tool.execute_with_context(
            {"session": source},
            ToolExecutionContext(source, (tmp_path / "other").resolve()),
        )
    with pytest.raises(ValueError, match="limit"):
        tool.execute_with_context(
            {"session": source, "limit": True},
            ToolExecutionContext(source, tmp_path.resolve()),
        )
    runtime.close()
