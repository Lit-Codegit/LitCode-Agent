from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from litcode_agent.agent import Agent
from litcode_agent.hooks import HookCommand, HookGroup, HookRunner, HookSettings
from litcode_agent.model import (
    AssistantTurn,
    Message,
    ModelError,
    ToolCall,
    ToolSchema,
)
from litcode_agent.tools.base import ToolResult
from litcode_agent.tools.registry import ToolRegistry


def test_hook_receives_json_and_respects_matcher(tmp_path: Path) -> None:
    settings = HookSettings(
        post_tool_use=(
            HookGroup(
                matcher="read_file",
                hooks=(
                    HookCommand(
                        "python -c 'import json,sys; print(json.load(sys.stdin)[\"tool_name\"])'"
                    ),
                ),
            ),
        )
    )
    runner = HookRunner(tmp_path, settings)
    payload = {
        "session_id": "session",
        "cwd": str(tmp_path),
        "hook_event_name": "PostToolUse",
        "tool_name": "read_file",
    }

    matched = runner.run("PostToolUse", payload, match_value="read_file")
    skipped = runner.run("PostToolUse", payload, match_value="run_command")

    assert matched.executions[0].stdout.strip() == "read_file"
    assert skipped.executions == ()


def test_pre_tool_exit_two_blocks_with_stderr(tmp_path: Path) -> None:
    settings = HookSettings(
        pre_tool_use=(
            HookGroup(
                matcher="run_command",
                hooks=(
                    HookCommand(
                        "python -c 'import sys; print(\"禁止执行\", file=sys.stderr); raise SystemExit(2)'"
                    ),
                ),
            ),
        )
    )

    outcome = HookRunner(tmp_path, settings).run(
        "PreToolUse",
        {"hook_event_name": "PreToolUse"},
        match_value="run_command",
    )

    assert outcome.blocked
    assert outcome.reason == "禁止执行"


class ScriptedModel:
    def __init__(self) -> None:
        self.requests: list[list[Message]] = []

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> AssistantTurn:
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            return AssistantTurn(
                None,
                (ToolCall("call", "touch", '{"path":"blocked.txt"}'),),
            )
        return AssistantTurn("已处理阻止结果")


class TouchTool:
    name = "touch"
    description = "create a file"
    input_schema = {"type": "object"}

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def execute(self, arguments: Mapping[str, object]) -> ToolResult:
        (self.workspace / "blocked.txt").touch()
        return ToolResult("created")


def test_agent_returns_pre_tool_block_to_model_without_running_tool(
    tmp_path: Path,
) -> None:
    hooks = HookSettings(
        pre_tool_use=(
            HookGroup(
                matcher="touch",
                hooks=(HookCommand("python -c 'raise SystemExit(2)'"),),
            ),
        )
    )
    model = ScriptedModel()

    result = Agent(
        model,
        ToolRegistry([TouchTool(tmp_path)]),
        3,
        hooks=HookRunner(tmp_path, hooks),
    ).run("创建文件")

    assert result.succeeded
    assert not (tmp_path / "blocked.txt").exists()
    tool_message = model.requests[1][-1]
    tool_result = json.loads(tool_message["content"])
    assert tool_result["ok"] is False
    assert "PreToolUse" in tool_result["content"]


def test_session_end_hook_runs_when_model_request_fails(tmp_path: Path) -> None:
    class FailingModel:
        def complete(self, messages, tools):
            raise ModelError("offline")

    hooks = HookSettings(
        session_end=(
            HookGroup(
                matcher="model_error",
                hooks=(
                    HookCommand(
                        "python -c 'import json,pathlib,sys; "
                        "pathlib.Path(\"reason.txt\").write_text("
                        "json.load(sys.stdin)[\"reason\"])'"
                    ),
                ),
            ),
        )
    )

    with pytest.raises(ModelError, match="offline"):
        Agent(
            FailingModel(),
            ToolRegistry([]),
            1,
            hooks=HookRunner(tmp_path, hooks),
        ).run("测试错误")

    assert (tmp_path / "reason.txt").read_text() == "model_error"
