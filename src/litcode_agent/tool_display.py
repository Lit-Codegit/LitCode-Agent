"""把工具调用压缩成适合 TUI 时间线的标题与摘要。"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping

from litcode_agent.model import ToolCall
from litcode_agent.tools.base import FileChange

TITLE_VALUE_LIMIT = 100
RESULT_SUMMARY_LIMIT = 1_200
SUBAGENT_PREVIEW_LIMIT = 240

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


def subagent_title(
    tool_call: ToolCall,
    status: str,
    model: str,
    elapsed_seconds: float,
    *,
    alias: str | None = None,
) -> str:
    """Render one stable subagent row without exposing invocation UUIDs."""

    arguments = _arguments(tool_call.arguments)
    profile = _value(arguments.get("agent")) or "general"
    name = f"子代理 {_value(alias)}" if alias else "子代理"
    model_label = _value(model) or "unknown"
    elapsed = max(0, int(elapsed_seconds))
    return f"{status} {name} [{profile}] · {model_label} · {elapsed}s"


def subagent_running_summary(tool_call: ToolCall, activity: str) -> str:
    """Show the bounded assignment plus the child's latest durable activity."""

    arguments = _arguments(tool_call.arguments)
    prompt = _preview(arguments.get("prompt"), SUBAGENT_PREVIEW_LIMIT) or "（未提供任务）"
    current = " ".join(activity.split()) or "正在创建子会话…"
    return f"任务：{prompt}\n\n当前：{current}"


def subagent_result_summary(
    tool_call: ToolCall,
    content: str,
    *,
    is_error: bool,
) -> tuple[str | None, str | None, bool, str]:
    """Return alias, invocation id and a bounded human-facing result summary."""

    arguments = _arguments(tool_call.arguments)
    prompt = _preview(arguments.get("prompt"), SUBAGENT_PREVIEW_LIMIT) or "（未提供任务）"
    payload = _json_object(content)
    alias = _optional_text(payload.get("alias"))
    invocation_id = _optional_text(payload.get("invocation_id"))
    output = _optional_text(payload.get("output"))
    if is_error:
        result = _truncate(content.strip() or "（无错误信息）", RESULT_SUMMARY_LIMIT)
        label = "失败"
    elif output:
        result = _truncate(" ".join(output.split()), RESULT_SUMMARY_LIMIT)
        label = "完成摘要"
    elif payload.get("background") is True:
        result = "子会话已在后台启动。"
        label = "当前"
    else:
        result = _truncate(content.strip() or "（无输出）", RESULT_SUMMARY_LIMIT)
        label = "完成摘要"
    return (
        alias,
        invocation_id,
        payload.get("background") is True,
        f"任务：{prompt}\n\n{label}：{result}",
    )


def subagent_completion_summary(
    tool_call: ToolCall,
    output: str,
    *,
    failed: bool = False,
) -> str:
    """Summarise a background invocation once the runtime reaches a terminal state."""

    arguments = _arguments(tool_call.arguments)
    prompt = _preview(arguments.get("prompt"), SUBAGENT_PREVIEW_LIMIT) or "（未提供任务）"
    label = "失败" if failed else "完成摘要"
    preview = _truncate(" ".join(output.split()) or "（无输出）", RESULT_SUMMARY_LIMIT)
    return f"任务：{prompt}\n\n{label}：{preview}"


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


def _json_object(raw: str) -> Mapping[str, object]:
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


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _preview(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return _truncate(" ".join(value.split()), limit)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n… 已省略 {len(text) - limit} 个字符 …\n"
    remaining = max(0, limit - len(marker))
    head = (remaining + 1) // 2
    tail = remaining // 2
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"
