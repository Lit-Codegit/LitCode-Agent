from litcode_agent.model import ToolCall
from litcode_agent.tool_display import (
    change_diff,
    change_result_summary,
    subagent_completion_summary,
    subagent_result_summary,
    subagent_running_summary,
    subagent_title,
    tool_result_summary,
    tool_title,
)
from litcode_agent.tools.base import FileChange


def test_tool_title_selects_key_arguments_without_large_patch_text() -> None:
    call = ToolCall(
        "call",
        "apply_patch",
        '{"path":"src/app.py","old_text":"old content",'
        '"new_text":"very large new content"}',
    )

    title = tool_title(call, "●")

    assert title == "● apply_patch · src/app.py · 修改"
    assert "old content" not in title
    assert "new content" not in title


def test_run_command_title_is_single_line_and_bounded() -> None:
    call = ToolCall(
        "call",
        "run_command",
        '{"command":"python -m pytest '
        + "x" * 200
        + '"}',
    )

    title = tool_title(call, "✓")

    assert "\n" not in title
    assert title.endswith("…")
    assert len(title) < 130


def test_tool_title_preserves_leading_dot_in_hidden_path() -> None:
    call = ToolCall(
        "call",
        "read_file",
        '{"path":".github/workflows/test.yml"}',
    )

    assert tool_title(call, "✓") == (
        "✓ read_file · .github/workflows/test.yml"
    )


def test_tool_result_summary_keeps_head_and_tail_with_omission_count() -> None:
    summary = tool_result_summary("A" * 1_000 + "B" * 1_000, False)

    assert summary.startswith("状态：成功\n\nAAAA")
    assert "已省略" in summary
    assert summary.endswith("BBBB")
    assert len(summary) <= 1_210


def test_change_diff_renders_creation_as_all_added_lines() -> None:
    change = FileChange("README.md", None, "line1\nline2\n", False)
    diff = change_diff(change)

    assert diff.startswith("--- README.md\n+++ README.md")
    assert "@@ -0,0 +1,2 @@" in diff
    assert "+line1" in diff
    assert "+line2" in diff


def test_change_diff_renders_edit_hunks_and_bounded_lines() -> None:
    change = FileChange("a.py", "old line\n", "old line\n" + "n" * 400 + "\n", True)
    diff = change_diff(change)

    assert "@@ -1 +1,2 @@" in diff
    assert " old line" in diff
    assert "+n n" not in diff
    assert all(len(line) <= 200 for line in diff.splitlines())
    assert "…" in diff


def test_change_diff_overview_replaces_large_changes() -> None:
    before = [f"before {index}" for index in range(700)]
    after = [f"after {index}" for index in range(700)]
    change = FileChange("big.py", "\n".join(before), "\n".join(after), True)
    diff = change_diff(change)

    assert "修改前 700 行 → 修改后 700 行" in diff
    assert diff.splitlines()[-1].startswith("修改前")


def test_change_result_summary_precedes_diff_for_cards() -> None:
    change = FileChange("b.py", "x", "y", True)
    summary = change_result_summary(change)

    assert summary.startswith("已修改 b.py")
    assert "--- b.py" in summary


def test_ask_user_title_counts_questions() -> None:
    call = ToolCall(
        "call",
        "ask_user",
        '{"questions":[{"question":"q","header":"h","options":[{"label":"a","description":"b"}]}]}',
    )

    assert tool_title(call, "✓") == "✓ ask_user · 1 个问题"


def test_subagent_display_keeps_model_task_and_bounded_completion() -> None:
    call = ToolCall(
        "call",
        "spawn_subagent",
        '{"prompt":"检查 测试\\n失败原因","agent":"explore","background":true}',
    )

    assert subagent_title(call, "⠋", "model-a", 8.9) == (
        "⠋ 子代理 [explore] · model-a · 8s"
    )
    assert subagent_running_summary(call, "正在调用工具 · search_files") == (
        "任务：检查 测试 失败原因\n\n当前：正在调用工具 · search_files"
    )

    alias, invocation_id, background, summary = subagent_result_summary(
        call,
        '{"alias":"260901-1432-ABC","invocation_id":"inv-1",'
        '"background":true}',
        is_error=False,
    )
    assert (alias, invocation_id, background) == ("260901-1432-ABC", "inv-1", True)
    assert "子会话已在后台启动" in summary

    completed = subagent_completion_summary(call, "定位到 fixture 没有初始化。")
    assert completed.endswith("完成摘要：定位到 fixture 没有初始化。")
