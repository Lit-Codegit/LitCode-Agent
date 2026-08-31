"""Constrained tools for listing, reading, searching, and editing files."""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Mapping

from litcode_agent.tools.base import FileChange, ToolError, ToolResult
from litcode_agent.tools.workspace import Workspace
from litcode_agent.read_scope import ReadScope


def _string_argument(
    arguments: Mapping[str, object], name: str, *, default: str | None = None
) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or not value:
        raise ToolError(f"{name} must be a non-empty string")
    return value


def _integer_argument(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ToolError(f"{name} must be between {minimum} and {maximum}")
    return value


def truncate_output(text: str, limit: int) -> str:
    """Keep useful context from both ends of oversized tool output."""

    if len(text) <= limit:
        return text
    marker = f"\n... output truncated ({len(text) - limit} characters omitted) ...\n"
    remaining = limit - len(marker)
    if remaining <= 0:
        return text[:limit]
    head = (remaining + 1) // 2
    tail = remaining // 2
    return f"{text[:head]}{marker}{text[-tail:]}"


class ListFilesTool:
    name = "list_files"
    description = "List files and directories below a workspace-relative path."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory"},
            "depth": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace | ReadScope, max_output_chars: int) -> None:
        self.scope = workspace if isinstance(workspace, ReadScope) else ReadScope(workspace)
        self.workspace = self.scope.workspace
        self.max_output_chars = max_output_chars

    def execute(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str) or not raw_path:
            raise ToolError("path must be a non-empty string")
        depth = _integer_argument(
            arguments, "depth", default=2, minimum=1, maximum=8
        )
        root = self.scope.resolve(raw_path).path
        if not root.is_dir():
            raise ToolError(f"path is not a directory: {raw_path}")

        lines: list[str] = []
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            relative_depth = len(current_path.relative_to(root).parts)
            directories.sort()
            files.sort()
            if relative_depth >= depth:
                directories.clear()
            for directory in directories:
                path = current_path / directory
                lines.append(f"{self.scope.display(path)}/")
            lines.extend(
                self.scope.display(current_path / filename) for filename in files
            )
        content = "\n".join(lines) or "(empty directory)"
        return ToolResult(truncate_output(content, self.max_output_chars))


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file with one-based line numbers."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Workspace | ReadScope, max_output_chars: int) -> None:
        self.scope = workspace if isinstance(workspace, ReadScope) else ReadScope(workspace)
        self.workspace = self.scope.workspace
        self.max_output_chars = max_output_chars

    def execute(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_path = _string_argument(arguments, "path")
        start = _integer_argument(
            arguments, "start_line", default=1, minimum=1, maximum=10_000_000
        )
        end = _integer_argument(
            arguments,
            "end_line",
            default=start + 399,
            minimum=1,
            maximum=10_000_000,
        )
        if end < start:
            raise ToolError("end_line must be greater than or equal to start_line")
        path = self.scope.resolve(raw_path).path
        if not path.is_file():
            raise ToolError(f"path is not a file: {raw_path}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ToolError(f"file is not valid UTF-8 text: {raw_path}") from error
        selected = lines[start - 1 : end]
        content = "\n".join(
            f"{number:>6} | {line}"
            for number, line in enumerate(selected, start=start)
        )
        if not content:
            content = f"(no lines in requested range; file has {len(lines)} lines)"
        return ToolResult(truncate_output(content, self.max_output_chars))


class SearchFilesTool:
    name = "search_files"
    description = "Search workspace text files with ripgrep."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def __init__(
        self, workspace: Workspace | ReadScope, max_output_chars: int, timeout_seconds: float
    ) -> None:
        self.scope = workspace if isinstance(workspace, ReadScope) else ReadScope(workspace)
        self.workspace = self.scope.workspace
        self.max_output_chars = max_output_chars
        self.timeout_seconds = timeout_seconds

    def execute(self, arguments: Mapping[str, object]) -> ToolResult:
        pattern = _string_argument(arguments, "pattern")
        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str) or not raw_path:
            raise ToolError("path must be a non-empty string")
        search_path = self.scope.resolve(raw_path).path
        command = ["rg", "--line-number", "--color", "never", "--", pattern]
        glob = arguments.get("glob")
        if glob is not None:
            if not isinstance(glob, str) or not glob:
                raise ToolError("glob must be a non-empty string")
            command[1:1] = ["--glob", glob]
        command.append(str(search_path))
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise ToolError("ripgrep (rg) is not installed") from error
        except subprocess.TimeoutExpired as error:
            raise ToolError(
                f"search timed out after {self.timeout_seconds:g} seconds"
            ) from error
        if completed.returncode == 1:
            return ToolResult("(no matches)")
        if completed.returncode != 0:
            raise ToolError(completed.stderr.strip() or "ripgrep failed")
        return ToolResult(truncate_output(completed.stdout, self.max_output_chars))


class ApplyPatchTool:
    name = "apply_patch"
    description = (
        "Apply one exact text replacement atomically. To create a file, provide "
        "an empty old_text and a non-empty new_text."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Workspace,
        execution_lock: threading.RLock | None = None,
    ) -> None:
        self.workspace = workspace
        self.execution_lock = execution_lock or threading.RLock()

    def execute(self, arguments: Mapping[str, object]) -> ToolResult:
        with self.execution_lock:
            return self._execute_locked(arguments)

    def _execute_locked(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_path = _string_argument(arguments, "path")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolError("old_text and new_text must be strings")
        path = self.workspace.resolve(raw_path, must_exist=False)

        if not path.exists():
            if old_text:
                raise ToolError("cannot replace text because the file does not exist")
            if not new_text:
                raise ToolError("new_text must not be empty when creating a file")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(path, new_text)
            return ToolResult(
                f"created {raw_path} ({len(new_text)} characters)",
                file_change=FileChange(raw_path, None, new_text, False),
            )

        if not path.is_file():
            raise ToolError(f"path is not a file: {raw_path}")
        if not old_text:
            raise ToolError("old_text must not be empty when editing an existing file")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError(f"file is not valid UTF-8 text: {raw_path}") from error
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise ToolError(
                f"old_text must match exactly once; found {occurrences} matches"
            )
        updated = content.replace(old_text, new_text, 1)
        self._atomic_write(path, updated)
        return ToolResult(
            f"updated {raw_path}",
            file_change=FileChange(raw_path, content, updated, True),
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        mode = path.stat().st_mode if path.exists() else None
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_name = temporary.name
            if mode is not None:
                os.chmod(temporary_name, mode)
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
