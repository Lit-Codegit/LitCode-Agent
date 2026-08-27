from io import StringIO

from rich.console import Console

from litcode_agent.agent import AgentEvent
from litcode_agent.model import ToolCall
from litcode_agent.ui import TerminalUI


def make_ui(*answers: str) -> tuple[TerminalUI, StringIO]:
    output = StringIO()
    inputs = iter(answers)
    return (
        TerminalUI(
            Console(
                file=output,
                color_system=None,
                force_terminal=False,
                width=100,
            ),
            lambda prompt: next(inputs),
        ),
        output,
    )


def test_choose_model_uses_numbered_selection() -> None:
    ui, output = make_ui("2")

    selected = ui.choose_model(("model-a", "model-b"), "model-a")

    assert selected == "model-b"
    assert "model-a" in output.getvalue()
    assert "model-b" in output.getvalue()


def test_renders_tool_call_and_result() -> None:
    ui, output = make_ui()
    call = ToolCall("call-1", "read_file", '{"path":"README.md"}')

    ui.handle_event(AgentEvent("tool_start", 1, tool_call=call))
    ui.handle_event(
        AgentEvent("tool_result", 1, tool_call=call, content="文件内容")
    )

    rendered = output.getvalue()
    assert "工具 · read_file" in rendered
    assert '"path": "README.md"' in rendered
    assert "工具结果 · 完成" in rendered
    assert "文件内容" in rendered


def test_invalid_model_selection_keeps_current_model() -> None:
    ui, output = make_ui("99")

    selected = ui.choose_model(("model-a",), "model-a")

    assert selected == "model-a"
    assert "模型序号超出范围" in output.getvalue()
