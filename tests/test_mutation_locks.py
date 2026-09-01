from __future__ import annotations

from pathlib import Path
import threading
import time

from litcode_agent.mutation_locks import WorkspaceMutationLocks, file_version
from litcode_agent.tools.base import ToolError
from litcode_agent.tools.files import ApplyPatchTool
from litcode_agent.tools.workspace import Workspace


def test_different_files_can_hold_write_locks_in_parallel(tmp_path: Path) -> None:
    locks = WorkspaceMutationLocks()
    entered = {"a": threading.Event(), "b": threading.Event()}
    release = threading.Event()

    def worker(name: str) -> None:
        with locks.write(tmp_path / f"{name}.txt"):
            entered[name].set()
            release.wait(1)

    first = threading.Thread(target=worker, args=("a",))
    second = threading.Thread(target=worker, args=("b",))
    first.start()
    second.start()
    assert entered["a"].wait(1)
    assert entered["b"].wait(1)
    release.set()
    first.join(1)
    second.join(1)


def test_command_lock_blocks_writes_and_releases_after_timeout_shape(tmp_path: Path) -> None:
    locks = WorkspaceMutationLocks()
    entered = threading.Event()
    release = threading.Event()
    write_finished = threading.Event()

    def command() -> None:
        with locks.command():
            entered.set()
            release.wait(1)

    def write() -> None:
        with locks.write(tmp_path / "shared.txt"):
            write_finished.set()

    command_thread = threading.Thread(target=command)
    command_thread.start()
    assert entered.wait(1)
    write_thread = threading.Thread(target=write)
    write_thread.start()
    time.sleep(0.05)
    assert not write_finished.is_set()
    release.set()
    command_thread.join(1)
    write_thread.join(1)
    assert write_finished.is_set()


def test_file_version_changes_only_with_content(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    assert file_version(target) is None
    target.write_text("one", encoding="utf-8")
    first = file_version(target)
    target.write_text("two", encoding="utf-8")
    assert first is not None
    assert file_version(target) != first


def test_apply_patch_reports_stale_file_version(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("one", encoding="utf-8")
    version = file_version(target)
    target.write_text("user edit", encoding="utf-8")
    try:
        ApplyPatchTool(Workspace(tmp_path)).execute(
            {
                "path": "file.txt",
                "old_text": "user edit",
                "new_text": "agent edit",
                "expected_version": version,
            }
        )
    except ToolError as error:
        assert "version conflict" in str(error)
    else:
        raise AssertionError("stale version should be rejected")
