from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import litcode_agent.process_runner as process_runner
from litcode_agent.process_runner import ShellSpec, resolve_shell
from litcode_agent.workspace_lock import WorkspaceProcessLock, WorkspaceLockError


def test_cli_import_does_not_require_unix_fcntl() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['fcntl'] = None; import litcode_agent.cli",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_workspace_process_lock_excludes_second_owner(tmp_path: Path) -> None:
    first = WorkspaceProcessLock.acquire(tmp_path)
    try:
        assert (tmp_path / ".litcode" / "runtime.lock").is_file()
        first.stream.seek(0)
        assert first.stream.read().strip()
        with pytest.raises(WorkspaceLockError):
            WorkspaceProcessLock.acquire(tmp_path)
    finally:
        first.release()

    second = WorkspaceProcessLock.acquire(tmp_path)
    second.release()


def test_unix_shell_keeps_existing_shell_true_behavior() -> None:
    spec = resolve_shell(platform_name="posix", environ={}, which=lambda _: None)

    assert spec == ShellSpec("/bin/sh", None)
    assert spec.invocation("pwd") == ("pwd", True)


def test_windows_shell_prefers_powershell_core() -> None:
    available = {"pwsh": "C:/Program Files/PowerShell/7/pwsh.exe"}
    spec = resolve_shell(
        platform_name="nt",
        environ={"COMSPEC": "C:/Windows/System32/cmd.exe"},
        which=lambda name: available.get(name),
    )

    assert spec.name == "PowerShell"
    assert spec.invocation("Get-Location") == (
        [
            "C:/Program Files/PowerShell/7/pwsh.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-Location\nexit $LASTEXITCODE",
        ],
        False,
    )


def test_windows_shell_falls_back_to_comspec() -> None:
    spec = resolve_shell(
        platform_name="nt",
        environ={"COMSPEC": "C:/Windows/System32/cmd.exe"},
        which=lambda _: None,
    )

    assert spec.name == "cmd.exe"
    assert spec.invocation("cd") == (
        ["C:/Windows/System32/cmd.exe", "/d", "/s", "/c", "cd"],
        False,
    )


def test_powershell_propagates_native_exit_code() -> None:
    """PowerShell 7.4+ -Command 默认不传播 $LASTEXITCODE，hook 依赖退出码
    2 表示拦截；显式 exit 包装后必须保留原退出码。"""

    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh not installed")
    spec = ShellSpec("PowerShell", pwsh)
    script = f"{sys.executable} -c 'import sys; sys.exit(2)'"
    invocation, implicit = spec.invocation(script)
    completed = subprocess.run(
        invocation,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )

    assert completed.returncode == 2


def test_windows_timeout_falls_back_when_taskkill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 42
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

    process = Process()
    monkeypatch.setattr(
        process_runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
    )

    process_runner._terminate_windows_tree(process)  # type: ignore[arg-type]

    assert process.killed
