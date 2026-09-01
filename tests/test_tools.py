from __future__ import annotations

import json
import subprocess
import threading
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
from litcode_agent.config import ReadRoot, Settings
from litcode_agent.read_scope import ReadScope
from litcode_agent.tools.base import ToolError, ToolExecutionContext
from litcode_agent.tools.command import is_dangerous_command
from litcode_agent.tools.files import truncate_output
from litcode_agent.session_store import SessionStore
from litcode_agent.orchestration import OrchestrationService


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


def test_default_registry_contains_core_and_progressive_context_tools(tmp_path: Path) -> None:
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
        "load_skill",
    }


def test_default_registry_serializes_workspace_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "example-model"},
    )
    registry = build_default_registry(settings)
    command_started = threading.Event()
    release_command = threading.Event()
    patch_finished = threading.Event()

    def blocking_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command_started.set()
        assert release_command.wait(2)
        return subprocess.CompletedProcess(args="noop", returncode=0, stdout="", stderr="")

    monkeypatch.setattr("litcode_agent.tools.command.subprocess.run", blocking_run)
    command_thread = threading.Thread(
        target=lambda: registry.execute_json(
            "run_command", json.dumps({"command": "noop"})
        )
    )
    patch_thread = threading.Thread(
        target=lambda: (
            registry.execute_json(
                "apply_patch",
                json.dumps(
                    {"path": "shared.txt", "old_text": "", "new_text": "done"}
                ),
            ),
            patch_finished.set(),
        )
    )

    command_thread.start()
    assert command_started.wait(1)
    patch_thread.start()
    assert not patch_finished.wait(0.05)
    assert not (tmp_path / "shared.txt").exists()
    release_command.set()
    command_thread.join(2)
    patch_thread.join(2)

    assert patch_finished.is_set()
    assert (tmp_path / "shared.txt").read_text() == "done"


def test_read_tools_can_use_named_external_root_but_patch_cannot(
    tmp_path: Path,
) -> None:
    workspace_path = tmp_path / "workspace"
    reference_path = tmp_path / "reference"
    workspace_path.mkdir()
    reference_path.mkdir()
    (reference_path / "notes.txt").write_text("external", encoding="utf-8")
    workspace = Workspace(workspace_path)
    scope = ReadScope(
        workspace, (ReadRoot("docs", reference_path.resolve(), False),)
    )

    read = ReadFileTool(scope, 1000).execute({"path": "docs/notes.txt"})

    assert "external" in read.content
    with pytest.raises(ToolError, match="does not exist"):
        ApplyPatchTool(workspace).execute(
            {"path": "docs/notes.txt", "old_text": "external", "new_text": "x"}
        )


def test_session_tools_use_runtime_source_identity_and_confirmation(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    target = store.create(tmp_path, "model", [])
    target_alias = store.session_info(target).alias
    confirmations: list[str] = []
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "model"},
    )
    registry = build_default_registry(
        settings,
        store=store,
        confirm_session_message=lambda description: not confirmations.append(description),
    )

    result = registry.execute_json(
        "send_session_message",
        json.dumps({"session": target_alias, "instruction": "运行测试"}),
        ToolExecutionContext(source, tmp_path.resolve()),
    )

    assert not result.is_error
    assert confirmations and target_alias in confirmations[0]
    assert store.inbox(target)[0].source_session_id == source


def test_list_sessions_exposes_same_terminal_pane_locations(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(
        tmp_path, "model", [], origin_terminal_id="T-NOW", origin_pane_slot=1
    )
    target = store.create(
        tmp_path, "model", [], origin_terminal_id="T-NOW", origin_pane_slot=2
    )
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "model"},
    )
    registry = build_default_registry(settings, store=store)

    result = registry.execute_json(
        "list_sessions",
        "{}",
        ToolExecutionContext(
            source,
            tmp_path.resolve(),
            terminal_id="T-NOW",
            pane_slot=1,
            mounted_sessions=((source, 1), (target, 2)),
        ),
    )

    payload = json.loads(result.content)
    assert [(item["scope"], item["pane"]) for item in payload[:2]] == [
        ("mounted", 1),
        ("mounted", 2),
    ]


def test_orchestration_tools_bind_source_and_reporter_to_runtime_identity(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    coordinator = store.create(tmp_path, "model", [])
    implementer = store.create(tmp_path, "model", [])
    service = OrchestrationService(store, tmp_path)
    run = service.start_run(coordinator, "实现功能")
    service.approve_run(run.id, coordinator)
    settings = Settings.from_env(
        tmp_path,
        {
            "OPENAI_API_KEY": "secret",
            "LITCODE_MODEL": "model",
            "LITCODE_SESSION_WAKE_POLICY": "allow",
        },
    )
    registry = build_default_registry(
        settings, store=store, orchestrator=service
    )

    delegated = registry.execute_json(
        "delegate_session",
        json.dumps(
            {
                "run_id": run.id,
                "session": store.session_info(implementer).alias,
                "role": "implementer",
                "objective": "修改 parser",
                "acceptance": ["测试通过"],
                "allowed_paths": ["src/parser.py"],
                "write_policy": "workspace-write",
            }
        ),
        ToolExecutionContext(coordinator, tmp_path.resolve()),
    )
    task_id = json.loads(delegated.content)["task_id"]
    service.start_task(task_id, implementer)
    reported = registry.execute_json(
        "report_task",
        json.dumps(
            {
                "task_id": task_id,
                "status": "completed",
                "summary": "完成",
                "evidence": ["pytest passed"],
                "changed_files": ["src/parser.py"],
            }
        ),
        ToolExecutionContext(implementer, tmp_path.resolve()),
    )

    assert not delegated.is_error
    assert not reported.is_error
    assert service.get_task(task_id).source_session_id == coordinator
    assert service.get_task(task_id).target_session_id == implementer


def test_orchestration_role_policy_blocks_reviewer_writes_and_path_escape(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "model"},
    )
    registry = build_default_registry(settings)

    reviewer = registry.execute_json(
        "apply_patch",
        json.dumps({"path": "review.txt", "old_text": "", "new_text": "x"}),
        ToolExecutionContext(
            "reviewer",
            tmp_path.resolve(),
            orchestration_role="reviewer",
            orchestration_write_policy="none",
        ),
    )
    outside_scope = registry.execute_json(
        "apply_patch",
        json.dumps({"path": "other.py", "old_text": "", "new_text": "x"}),
        ToolExecutionContext(
            "implementer",
            tmp_path.resolve(),
            orchestration_role="implementer",
            orchestration_write_policy="workspace-write",
            orchestration_allowed_paths=("src/parser.py",),
        ),
    )

    assert reviewer.is_error and "reviewer" in reviewer.content
    assert outside_scope.is_error and "allowed_paths" in outside_scope.content


def test_session_context_tool_returns_query_bounded_excerpt(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    store.save_messages(
        source,
        [{"role": "assistant", "content": "测试结论：需要 workspace lock"}],
    )
    alias = store.session_info(source).alias
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "model"},
    )
    registry = build_default_registry(settings, store=store)

    result = registry.execute_json(
        "read_session_context",
        json.dumps({"session": alias, "query": "workspace", "max_chars": 64}),
        ToolExecutionContext(source, tmp_path.resolve()),
    )

    assert "workspace lock" in result.content
    assert len(result.content) <= 64


def test_session_read_policy_can_disable_model_session_inspection(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    settings = Settings.from_env(
        tmp_path,
        {
            "OPENAI_API_KEY": "secret",
            "LITCODE_MODEL": "model",
            "LITCODE_SESSION_READ_POLICY": "deny",
        },
    )
    registry = build_default_registry(settings, store=store)

    result = registry.execute_json(
        "list_sessions",
        "{}",
        ToolExecutionContext(source, tmp_path.resolve()),
    )

    assert result.is_error
    assert "denied" in result.content
