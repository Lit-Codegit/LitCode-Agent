from litcode_agent.model import ToolCall
from litcode_agent.tool_display import tool_result_summary, tool_title


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
