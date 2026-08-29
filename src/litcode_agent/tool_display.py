"""把工具调用压缩成适合 TUI 时间线的标题与摘要。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from litcode_agent.model import ToolCall

TITLE_VALUE_LIMIT = 100
RESULT_SUMMARY_LIMIT = 1_200


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
