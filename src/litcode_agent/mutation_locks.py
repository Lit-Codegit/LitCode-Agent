"""Short-lived workspace mutation locks.

Reads and model thinking remain unconstrained.  A command owns the workspace
lock for its subprocess lifetime; a file writer owns a real-path lock only
while checking its version and atomically replacing the file.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
import hashlib
import threading
from typing import Protocol


class MutationLocks(Protocol):
    """The narrow lock surface shared by file and command tools."""

    def command(self) -> AbstractContextManager[None]: ...

    def write(self, path: Path) -> AbstractContextManager[None]: ...

    def write_many(self, paths: Iterable[Path]) -> AbstractContextManager[None]: ...


class WorkspaceMutationLocks:
    """Coordinate commands and file writes without serialising different files."""

    _workspace_guard = threading.RLock()
    _workspace_instances: dict[Path, WorkspaceMutationLocks] = {}

    @classmethod
    def for_workspace(cls, workspace: Path) -> WorkspaceMutationLocks:
        """Return the process-wide lock shared by direct tool instances."""

        root = workspace.resolve()
        with cls._workspace_guard:
            return cls._workspace_instances.setdefault(root, cls())

    def __init__(self) -> None:
        self._state = threading.Condition(threading.RLock())
        self._command_active = False
        self._writers = 0
        self._file_locks: dict[Path, threading.RLock] = {}
        self._file_locks_guard = threading.RLock()

    @contextmanager
    def command(self) -> Iterator[None]:
        with self._state:
            while self._command_active or self._writers:
                self._state.wait()
            self._command_active = True
        try:
            yield
        finally:
            with self._state:
                self._command_active = False
                self._state.notify_all()

    @contextmanager
    def write(self, path: Path) -> Iterator[None]:
        with self._state:
            while self._command_active:
                self._state.wait()
            self._writers += 1
        lock = self._path_lock(path)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._state:
                self._writers -= 1
                self._state.notify_all()

    @contextmanager
    def write_many(self, paths: Iterable[Path]) -> Iterator[None]:
        """Acquire real-path locks in deterministic order to avoid deadlocks."""

        canonical = sorted({path.resolve(strict=False) for path in paths})
        with self._state:
            while self._command_active:
                self._state.wait()
            self._writers += 1
        locks = [self._path_lock(path) for path in canonical]
        for lock in locks:
            lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()
            with self._state:
                self._writers -= 1
                self._state.notify_all()

    @property
    def command_active(self) -> bool:
        with self._state:
            return self._command_active

    def _path_lock(self, path: Path) -> threading.RLock:
        canonical = path.resolve(strict=False)
        with self._file_locks_guard:
            return self._file_locks.setdefault(canonical, threading.RLock())


def file_version(path: Path) -> str | None:
    """Return a content hash, or ``None`` for a missing file."""

    if not path.exists():
        return None
    if not path.is_file():
        raise IsADirectoryError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
