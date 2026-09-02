"""Cross-platform shell selection and bounded command execution."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ShellSpec:
    """Describe the shell syntax exposed to the model and its invocation."""

    name: str
    executable: str | None

    def invocation(self, command: str) -> tuple[str | list[str], bool]:
        if self.executable is None:
            return command, True
        if self.name == "PowerShell":
            # PowerShell 7.4+ -Command 只把“最后一个语句”的成败映射为进程
            # 退出码（失败一律 1），不会透传原生命令的 $LASTEXITCODE；显式
            # exit 才能保留真实退出码（hook 依赖 return code == 2 的约定）。
            return (
                [
                    self.executable,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"{command}\nexit $LASTEXITCODE",
                ],
                False,
            )
        return [self.executable, "/d", "/s", "/c", command], False


def resolve_shell(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> ShellSpec:
    """Keep the Unix shell path stable and choose an explicit Windows shell."""

    target = os.name if platform_name is None else platform_name
    if target != "nt":
        return ShellSpec("/bin/sh", None)
    values = os.environ if environ is None else environ
    for candidate in ("pwsh", "powershell"):
        executable = which(candidate)
        if executable:
            return ShellSpec("PowerShell", executable)
    executable = values.get("COMSPEC") or which("cmd.exe") or "cmd.exe"
    return ShellSpec("cmd.exe", executable)


def run_shell(
    command: str,
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, terminating its Windows process tree on timeout."""

    spec = resolve_shell()
    invocation, use_implicit_shell = spec.invocation(command)
    if os.name != "nt":
        # This is deliberately the pre-Windows implementation: preserving it
        # avoids changing quoting, environment expansion, and signal behaviour
        # for existing Unix users.
        return subprocess.run(
            invocation,
            cwd=cwd,
            shell=use_implicit_shell,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        invocation,
        cwd=cwd,
        shell=False,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _terminate_windows_tree(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            error.cmd,
            error.timeout,
            output=error.output or stdout,
            stderr=error.stderr or stderr,
        ) from error
    return subprocess.CompletedProcess(invocation, process.returncode, stdout, stderr)


def _terminate_windows_tree(process: subprocess.Popen[str]) -> None:
    taskkill: Sequence[str] = ("taskkill", "/pid", str(process.pid), "/t", "/f")
    try:
        completed = subprocess.run(
            taskkill,
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0 and process.poll() is None:
            process.kill()
    except OSError:
        process.kill()
