from __future__ import annotations

import json
from pathlib import Path

import pytest

from litcode_agent.tools import (
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    SearchFilesTool,
    ToolRegistry,
    Workspace,
    build_default_registry,
)
from litcode_agent.config import Settings
from litcode_agent.tools.base import ToolError
from litcode_agent.tools.command import is_dangerous_command
from litcode_agent.tools.files import truncate_output


def test_workspace_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(ToolError, match="escapes"):
        workspace.resolve("../outside.txt", must_exist=False)


def test_workspace_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="relative"):
        Workspace(tmp_path).resolve(str(tmp_path / "file.txt"), must_exist=False)


def test_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ToolError, match="escapes"):
        Workspace(tmp_path).resolve("escape/new.txt", must_exist=False)


def test_list_and_read_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "hello.py").write_text("one\ntwo\nthree\n")
    workspace = Workspace(tmp_path)

    listing = ListFilesTool(workspace, 2_000).execute({"depth": 2})
    reading = ReadFileTool(workspace, 2_000).execute(
        {"path": "src/hello.py", "start_line": 2, "end_line": 3}
    )

    assert "src/hello.py" in listing.content
    assert "2 | two" in reading.content
    assert "3 | three" in reading.content


def test_search_files_uses_workspace_relative_results(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("needle = 1\n")

    result = SearchFilesTool(Workspace(tmp_path), 2_000, 2).execute(
        {"pattern": "needle", "glob": "*.py"}
    )

    assert "hello.py:1:needle = 1" in result.content


def test_apply_patch_creates_and_edits_file(tmp_path: Path) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))

    created = tool.execute(
        {"path": "src/app.py", "old_text": "", "new_text": "answer = 41\n"}
    )
    updated = tool.execute(
        {
            "path": "src/app.py",
            "old_text": "answer = 41",
            "new_text": "answer = 42",
        }
    )

    assert created.is_error is False
    assert updated.is_error is False
    assert (tmp_path / "src" / "app.py").read_text() == "answer = 42\n"


def test_apply_patch_requires_one_exact_match(tmp_path: Path) -> None:
    (tmp_path / "values.txt").write_text("same\nsame\n")

    with pytest.raises(ToolError, match="found 2 matches"):
        ApplyPatchTool(Workspace(tmp_path)).execute(
            {"path": "values.txt", "old_text": "same", "new_text": "different"}
        )


@pytest.mark.parametrize(
    "command",
    [
        "rm file.txt",
        "/bin/rm file.txt",
        "env rm file.txt",
        "echo ok; rm file.txt",
        "sudo whoami",
        "git reset --hard HEAD~1",
        "git push origin main",
        "chmod -R 777 .",
    ],
)
def test_detects_dangerous_commands(command: str) -> None:
    assert is_dangerous_command(command)


def test_runs_safe_command(tmp_path: Path) -> None:
    result = RunCommandTool(Workspace(tmp_path), 2, 2_000, "deny").execute(
        {"command": "pwd"}
    )

    assert "exit_code: 0" in result.content
    assert str(tmp_path) in result.content


def test_denies_dangerous_command_without_running_it(tmp_path: Path) -> None:
    target = tmp_path / "keep.txt"
    target.write_text("keep")

    with pytest.raises(ToolError, match="denied"):
        RunCommandTool(Workspace(tmp_path), 2, 2_000, "deny").execute(
            {"command": "rm keep.txt"}
        )

    assert target.exists()


def test_confirmation_policy_controls_dangerous_command(tmp_path: Path) -> None:
    requested: list[str] = []
    tool = RunCommandTool(
        Workspace(tmp_path),
        2,
        2_000,
        "confirm",
        confirm=lambda command: bool(requested.append(command)),
    )

    with pytest.raises(ToolError, match="not approved"):
        tool.execute({"command": "rm missing.txt"})

    assert requested == ["rm missing.txt"]


def test_command_timeout_is_reported(tmp_path: Path) -> None:
    tool = RunCommandTool(Workspace(tmp_path), 0.01, 2_000, "deny")

    with pytest.raises(ToolError, match="timed out"):
        tool.execute({"command": "sleep 1"})


def test_output_truncation_keeps_both_ends() -> None:
    output = truncate_output("a" * 100 + "z" * 100, 100)

    assert len(output) == 100
    assert output.startswith("a")
    assert output.endswith("z")
    assert "truncated" in output


def test_registry_exports_schemas_and_reports_recoverable_errors(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry([ReadFileTool(Workspace(tmp_path), 2_000)])

    schemas = registry.schemas()
    unknown = registry.execute_json("missing", "{}")
    malformed = registry.execute_json("read_file", "not json")
    invalid = registry.execute_json("read_file", json.dumps({"path": "missing"}))

    assert schemas[0]["function"]["name"] == "read_file"  # type: ignore[index]
    assert unknown.is_error and "unknown tool" in unknown.content
    assert malformed.is_error and "invalid JSON" in malformed.content
    assert invalid.is_error and "does not exist" in invalid.content


def test_default_registry_contains_exactly_the_mvp_tools(tmp_path: Path) -> None:
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "example-model"},
    )

    names = {
        schema["function"]["name"]  # type: ignore[index]
        for schema in build_default_registry(settings).schemas()
    }

    assert names == {
        "list_files",
        "read_file",
        "search_files",
        "apply_patch",
        "run_command",
    }
