"""Workspace path confinement shared by filesystem tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from litcode_agent.tools.base import ToolError


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"workspace is not a directory: {root}")
        object.__setattr__(self, "root", root)

    def resolve(self, raw_path: str, *, must_exist: bool = True) -> Path:
        """Resolve a relative path and reject traversal or symlink escapes."""

        path = Path(raw_path)
        if path.is_absolute():
            raise ToolError("path must be relative to the workspace")
        resolved = (self.root / path).resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ToolError("path escapes the workspace")
        if must_exist and not resolved.exists():
            raise ToolError(f"path does not exist: {raw_path}")
        return resolved

    def display(self, path: Path) -> str:
        return "." if path == self.root else path.relative_to(self.root).as_posix()
