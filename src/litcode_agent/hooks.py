"""Claude Code 风格的本地命令 hook。"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

HookEvent = Literal[
    "SessionStart",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "SessionEnd",
]


@dataclass(frozen=True, slots=True)
class HookCommand:
    command: str
    timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class HookGroup:
    matcher: str
    hooks: tuple[HookCommand, ...]


@dataclass(frozen=True, slots=True)
class HookSettings:
    disabled: bool = False
    session_start: tuple[HookGroup, ...] = ()
    pre_tool_use: tuple[HookGroup, ...] = ()
    post_tool_use: tuple[HookGroup, ...] = ()
    post_tool_use_failure: tuple[HookGroup, ...] = ()
    session_end: tuple[HookGroup, ...] = ()

    def groups_for(self, event: HookEvent) -> tuple[HookGroup, ...]:
        return {
            "SessionStart": self.session_start,
            "PreToolUse": self.pre_tool_use,
            "PostToolUse": self.post_tool_use,
            "PostToolUseFailure": self.post_tool_use_failure,
            "SessionEnd": self.session_end,
        }[event]

    @property
    def count(self) -> int:
        return sum(
            len(group.hooks)
            for event in (
                "SessionStart",
                "PreToolUse",
                "PostToolUse",
                "PostToolUseFailure",
                "SessionEnd",
            )
            for group in self.groups_for(event)
        )


@dataclass(frozen=True, slots=True)
class HookExecution:
    event: HookEvent
    command: str
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class HookOutcome:
    executions: tuple[HookExecution, ...] = ()
    blocked: bool = False
    reason: str | None = None


class HookRunner:
    def __init__(self, workspace: Path, settings: HookSettings) -> None:
        self.workspace = workspace.resolve()
        self.settings = settings

    def run(
        self,
        event: HookEvent,
        payload: Mapping[str, object],
        *,
        match_value: str = "",
    ) -> HookOutcome:
        if self.settings.disabled:
            return HookOutcome()
        hook_input = json.dumps(payload, ensure_ascii=False)
        executions: list[HookExecution] = []
        blocked_reasons: list[str] = []
        for group in self.settings.groups_for(event):
            if group.matcher and re.fullmatch(group.matcher, match_value) is None:
                continue
            for hook in group.hooks:
                execution = self._execute(event, hook, hook_input)
                executions.append(execution)
                if event == "PreToolUse" and execution.return_code == 2:
                    blocked_reasons.append(
                        execution.stderr.strip() or "PreToolUse hook 已阻止工具调用"
                    )
        return HookOutcome(
            executions=tuple(executions),
            blocked=bool(blocked_reasons),
            reason="\n".join(blocked_reasons) or None,
        )

    def _execute(
        self, event: HookEvent, hook: HookCommand, hook_input: str
    ) -> HookExecution:
        command = hook.command.replace(
            "${LITCODE_PROJECT_DIR}", str(self.workspace)
        )
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                input=hook_input,
                capture_output=True,
                text=True,
                timeout=hook.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return HookExecution(
                event=event,
                command=command,
                return_code=-1,
                stdout=_decode_timeout_output(error.stdout),
                stderr=(
                    _decode_timeout_output(error.stderr)
                    or f"hook 在 {hook.timeout_seconds:g} 秒后超时"
                ),
                timed_out=True,
            )
        return HookExecution(
            event=event,
            command=command,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _decode_timeout_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    return output.decode(errors="replace") if isinstance(output, bytes) else output
