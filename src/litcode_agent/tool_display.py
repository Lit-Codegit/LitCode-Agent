"""把工具调用压缩成适合 TUI 时间线的标题与摘要。"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping

from litcode_agent.model import ToolCall
from litcode_agent.tools.base import FileChange

TITLE_VALUE_LIMIT = 100
RESULT_SUMMARY_LIMIT = 1_200

DIFF_CONTEXT_LINES = 3
DIFF_LINE_LIMIT = 600
DIFF_CHAR_LIMIT = 200


def tool_title(tool_call: ToolCall, status: str) -> str:
    """返回状态、工具名和稳定的关键参数。"""

    arguments = _arguments(tool_call.arguments)
    detail = _key_arguments(tool_call.name, arguments)
    suffix = f" · {detail}" if detail else ""
    return f"{status} {tool_call.name}{suffix}"


def tool_result_summary(content: str, is_error: bool) -> str:
    """展开卡片时只显示有界运行摘要。"""

    label = "失败" if is_error else "成功"
    normalized = content.strip() or "（无输出）"
    return f"状态：{label}\n\n{_truncate(normalized, RESULT_SUMMARY_LIMIT)}"


def change_result_summary(change: FileChange) -> str:
    """apply_patch 的结果展示为统一 diff；新建文件时即新增视图。"""

    verb = "已创建" if not change.before_exists else "已修改"
    diff = change_diff(change)
    return f"{verb} {change.path}\n\n{diff}"


def change_diff(change: FileChange) -> str:
    """为一次 apply_patch 生成有界统一 diff。

    difflib 对大输入较慢，因此只在前后文件都足够小的时候计算真实 diff；
    超大改动退化为行数统计与有界前后片段。
    """

    before = (change.before_content or "").splitlines()
    after = change.after_content.splitlines()
    if not change.before_exists or not change.before_content:
        return _creation_diff(change.path, after)
    if len(before) > DIFF_LINE_LIMIT or len(after) > DIFF_LINE_LIMIT:
        return _overview_diff(change.path, before, after)
    return _unified_diff(change.path, before, after)


def _unified_diff(path: str, before: list[str], after: list[str]) -> str:
    lines = list(
        difflib.unified_diff(
            before,
            after,
            fromfile=path,
            tofile=path,
            lineterm="",
            n=DIFF_CONTEXT_LINES,
        )
    )
    if len(lines) <= DIFF_LINE_LIMIT:
        return "\n".join(_bounded_line(line) for line in lines)
    head = 2 * (DIFF_LINE_LIMIT // 2)
    tail = DIFF_LINE_LIMIT - head
    ellipsis = f"\n… 已省略 {len(lines) - DIFF_LINE_LIMIT} 行 diff …\n"
    return (
        "\n".join(_bounded_line(line) for line in lines[:head])
        + ellipsis
        + "\n".join(_bounded_line(line) for line in lines[-tail:])
    )


def _creation_diff(path: str, after: list[str]) -> str:
    if len(after) <= DIFF_LINE_LIMIT:
        lines = list(
            difflib.unified_diff(
                [], after, fromfile=path, tofile=path, lineterm="", n=DIFF_CONTEXT_LINES
            )
        )
    else:
        head = 2 * (DIFF_LINE_LIMIT // 3)
        tail = DIFF_LINE_LIMIT - head
        lines = [f"--- {path}", f"+++ {path}", f"@@ -0,0 +1,{len(after)} @@"]
        lines.extend(f"+{line}" for line in after[:head])
        lines.append(f"… 已省略 {len(after) - head - tail} 行 …")
        lines.extend(f"+{line}" for line in after[-tail:])
    return "\n".join(_bounded_line(line) for line in lines)


def _overview_diff(path: str, before: list[str], after: list[str]) -> str:
    added = max(0, len(after) - len(before))
    removed = max(0, len(before) - len(after))
    return (
        f"--- {path}\n"
        f"+++ {path}\n"
        f"@@ 修改行数超过上限，diff 省略 @@\n"
        f"修改前 {len(before)} 行 → 修改后 {len(after)} 行（+{added} / -{removed}）"
    )


def _bounded_line(line: str) -> str:
    if len(line) <= DIFF_CHAR_LIMIT:
        return line
    return f"{line[: DIFF_CHAR_LIMIT - 1]}…"


def _arguments(raw: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _key_arguments(name: str, arguments: Mapping[str, object]) -> str:
    if name == "read_file":
        path = _value(arguments.get("path"))
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        lines = ""
        if isinstance(start, int):
            lines = f" · {start}–{end}" if isinstance(end, int) else f" · {start}+"
        return f"{path}{lines}"
    if name == "list_files":
        path = _value(arguments.get("path", "."))
        depth = arguments.get("depth")
        return f"{path} · depth={depth}" if isinstance(depth, int) else path
    if name == "search_files":
        pattern = _quoted(arguments.get("pattern"))
        path = _value(arguments.get("path", "."))
        glob = arguments.get("glob")
        detail = _join(pattern, path)
        return _join(detail, _value(glob)) if glob else detail
    if name == "apply_patch":
        path = _value(arguments.get("path"))
        action = "创建" if arguments.get("old_text") == "" else "修改"
        return _join(path, action)
    if name == "run_command":
        return _value(arguments.get("command"))
    if name == "ask_user":
        questions = arguments.get("questions")
        count = len(questions) if isinstance(questions, list) else 0
        return f"{count} 个问题"

    values = [
        f"{key}={_value(value)}"
        for key, value in arguments.items()
        if isinstance(value, (str, int, float, bool))
    ]
    return " · ".join(values[:2])


def _quoted(value: object) -> str:
    text = _value(value)
    return f'"{text}"' if text else ""


def _join(*parts: str) -> str:
    return " · ".join(part for part in parts if part)


def _value(value: object) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= TITLE_VALUE_LIMIT:
        return text
    return f"{text[: TITLE_VALUE_LIMIT - 1]}…"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n… 已省略 {len(text) - limit} 个字符 …\n"
    remaining = max(0, limit - len(marker))
    head = (remaining + 1) // 2
    tail = remaining // 2
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"
