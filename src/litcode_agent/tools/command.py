"""Command execution with explicit limits and dangerous-command policy."""

from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Mapping

from litcode_agent.config import CommandPolicy
from litcode_agent.tools.base import ToolError, ToolResult, UserDeclinedError
from litcode_agent.tools.files import _string_argument, truncate_output
from litcode_agent.mutation_locks import MutationLocks, WorkspaceMutationLocks
from litcode_agent.process_runner import run_shell
from litcode_agent.tools.workspace import Workspace

if TYPE_CHECKING:
    from litcode_agent.session_runtime import SessionRuntime

ConfirmCommand = Callable[[str], bool]

_DANGEROUS_PATTERNS = (
    re.compile(r"\brm(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(?:push|branch\s+-D)\b", re.IGNORECASE),
    re.compile(r"\b(?:mkfs|shutdown|reboot)\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+-R\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b", re.IGNORECASE),
    re.compile(r"\b(?:del|erase|rd|rmdir)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:stop-computer|restart-computer|format|diskpart)\b", re.IGNORECASE
    ),
)


def is_dangerous_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in _DANGEROUS_PATTERNS)


class RunCommandTool:
    name = "run_command"
    description = (
        "Run a shell command from the workspace with a timeout. This is not an "
        "operating-system sandbox; dangerous commands are controlled by policy."
    )
    input_schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Workspace,
        timeout_seconds: float,
        max_output_chars: int,
        policy: CommandPolicy,
        confirm: ConfirmCommand | None = None,
        execution_lock: MutationLocks | None = None,
        runtime: SessionRuntime | None = None,
    ) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.policy = policy
        self.confirm = confirm
        self.runtime = runtime
        self.execution_lock = execution_lock or WorkspaceMutationLocks.for_workspace(
            workspace.root
        )

    def execute(self, arguments: Mapping[str, object]) -> ToolResult:
        command = _string_argument(arguments, "command")
        if is_dangerous_command(command):
            if self.policy == "deny":
                raise ToolError("dangerous command denied by policy")
            if self.confirm is None:
                approved = False
            elif self.runtime is not None:
                approved = self.runtime.request_confirmation(
                    lambda: self.confirm(command)
                )
            else:
                approved = self.confirm(command)
            if self.policy == "confirm" and not approved:
                raise UserDeclinedError("dangerous command was not approved")
        try:
            with self.execution_lock.command():
                completed = run_shell(
                    command,
                    cwd=self.workspace.root,
                    timeout=self.timeout_seconds,
                )
        except subprocess.TimeoutExpired as error:
            partial_stdout = error.stdout or ""
            partial_stderr = error.stderr or ""
            if isinstance(partial_stdout, bytes):
                partial_stdout = partial_stdout.decode(errors="replace")
            if isinstance(partial_stderr, bytes):
                partial_stderr = partial_stderr.decode(errors="replace")
            details = truncate_output(
                f"stdout:\n{partial_stdout}\nstderr:\n{partial_stderr}",
                self.max_output_chars,
            )
            raise ToolError(
                f"command timed out after {self.timeout_seconds:g} seconds\n{details}"
            ) from error

        content = (
            f"exit_code: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        return ToolResult(truncate_output(content, self.max_output_chars))
