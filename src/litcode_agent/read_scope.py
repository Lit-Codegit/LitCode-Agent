"""Read-only named roots kept separate from the writable workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from litcode_agent.config import ReadRoot
from litcode_agent.tools.base import ToolError
from litcode_agent.tools.workspace import Workspace


@dataclass(frozen=True, slots=True)
class ResolvedReadPath:
    path: Path
    display: str
    external: bool


class ReadScope:
    def __init__(self, workspace: Workspace, roots: tuple[ReadRoot, ...] = ()) -> None:
        self.workspace = workspace
        self.roots = {root.alias: root for root in roots}

    def resolve(self, raw_path: str, *, must_exist: bool = True) -> ResolvedReadPath:
        normalized = raw_path.removeprefix("@")
        alias, separator, rest = normalized.partition("/")
        root = self.roots.get(alias) if separator else None
        if root is None:
            path = self.workspace.resolve(raw_path, must_exist=must_exist)
            return ResolvedReadPath(path, self.workspace.display(path), False)
        candidate = (root.path / rest).resolve(strict=False)
        if not candidate.is_relative_to(root.path):
            raise ToolError("path escapes the configured read root")
        if must_exist and not candidate.exists():
            raise ToolError(f"path does not exist: {raw_path}")
        display = alias if not rest else f"{alias}/{Path(rest).as_posix()}"
        return ResolvedReadPath(candidate, display, True)

    def display(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        if resolved.is_relative_to(self.workspace.root):
            return self.workspace.display(resolved)
        for alias, root in self.roots.items():
            if resolved.is_relative_to(root.path):
                relative = resolved.relative_to(root.path).as_posix()
                return alias if relative == "." else f"{alias}/{relative}"
        raise ToolError("path is outside readable roots")
