"""Single-process workspace lock with native Unix and Windows backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class WorkspaceLockError(RuntimeError):
    """The workspace lock is already held or cannot be acquired."""


@dataclass(slots=True)
class WorkspaceProcessLock:
    stream: BinaryIO
    backend: str
    _released: bool = False

    @classmethod
    def acquire(cls, workspace: Path) -> WorkspaceProcessLock:
        lock_path = workspace / ".litcode" / "runtime.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+b")
        backend = "windows" if os.name == "nt" else "unix"
        try:
            if backend == "windows":
                import msvcrt

                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as error:
            stream.close()
            raise WorkspaceLockError("workspace lock is already held") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()).encode("ascii"))
        stream.flush()
        return cls(stream, backend)

    def release(self) -> None:
        if self._released:
            return
        try:
            if self.backend == "windows":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self._released = True
