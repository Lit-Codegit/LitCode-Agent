from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from textual.widgets import Collapsible, Markdown, Static

from litcode_agent.config import Settings
from litcode_agent.model import AssistantTurn, ModelDelta, ToolCall
from litcode_agent.tui import (
    COMMANDS,
    ConfirmCommand,
    LitCodeTUI,
    ModelPicker,
    PromptArea,
)


class FakeModel:
    def __init__(self) -> None:
        self.model = "model-a"
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(list(messages))
        return AssistantTurn("TUI 回答")

    def list_models(self):
        return ("model-a", "model-b")

    def select_model(self, model):
        self.model = model


def settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "model-a"},
    )


def test_tui_mounts_status_timeline_and_fixed_prompt(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)):
            assert "model-a" in str(app.query_one("#status", Static).render())
            assert app.query_one(PromptArea).has_focus
            assert any(
                "会话已启动" in str(widget.render())
                for widget in app.query(".notice").results(Static)
            )

    asyncio.run(exercise())


def test_tui_submits_prompt_in_background_and_renders_answer(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "检查项目"
            await pilot.press("ctrl+enter")
            for _ in range(20):
                await pilot.pause(0.02)
                if not app.busy and model.requests:
                    break

            assert model.requests[0][-1] == {
                "role": "user",
                "content": "检查项目",
            }
            markdown = list(app.query(Markdown))
            assert len(markdown) == 1
            assert not app.busy

    asyncio.run(exercise())


def test_tui_model_picker_switches_current_model(tmp_path: Path) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("f2")
            for _ in range(20):
                await pilot.pause(0.02)
                if isinstance(app.screen, ModelPicker):
                    break
            assert isinstance(app.screen, ModelPicker)

            await pilot.press("down", "enter")
            await pilot.pause()

            assert model.model == "model-b"

    asyncio.run(exercise())


def test_tui_denies_dangerous_command_in_modal(tmp_path: Path) -> None:
    class ToolModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return AssistantTurn(
                    None,
                    (ToolCall("danger", "run_command", '{"command":"git push"}'),),
                )
            return AssistantTurn("已收到拒绝结果")

    async def exercise() -> None:
        model = ToolModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "执行危险命令"
            await pilot.press("ctrl+enter")
            for _ in range(30):
                await pilot.pause(0.02)
                if isinstance(app.screen, ConfirmCommand):
                    break
            assert isinstance(app.screen, ConfirmCommand)

            await pilot.press("escape")
            for _ in range(30):
                await pilot.pause(0.02)
                if not app.busy and len(model.requests) == 2:
                    break

            tool_message = model.requests[1][-1]
            assert "dangerous command was not approved" in tool_message["content"]
            assert not app.busy

    asyncio.run(exercise())


def test_slash_command_uses_fuzzy_inline_completion(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "/mo"
            prompt.move_cursor((0, 3))
            await pilot.pause()

            assert app.completion_visible
            assert "/model" in app.completion_values
            await pilot.press("enter")

            assert prompt.text == "/model "
            assert not app.completion_visible

    asyncio.run(exercise())


def test_command_registry_includes_session_workflows() -> None:
    assert {item.name for item in COMMANDS} >= {
        "/sessions",
        "/compact",
        "/rewind",
        "/redo",
        "/fork",
    }


def test_tui_manual_compaction_keeps_session_available(tmp_path: Path) -> None:
    class CompactModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            return AssistantTurn("摘要" if not tools else "回答")

    async def exercise() -> None:
        model = CompactModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "问题"
            await pilot.press("ctrl+enter")
            for _ in range(30):
                await pilot.pause(0.02)
                if not app.busy:
                    break
            app._handle_command("/compact")
            for _ in range(30):
                await pilot.pause(0.02)
                if not app.busy:
                    break

            assert app.session.summary is not None
            assert app.session.summary[0] == "摘要"

    asyncio.run(exercise())


def test_at_completion_references_workspace_file_in_model_prompt(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("print('引用内容')", encoding="utf-8")

    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(30):
                await pilot.pause(0.02)
                if "main.py" in app.file_paths:
                    break
            prompt = app.query_one(PromptArea)
            prompt.text = "检查 @ma"
            prompt.move_cursor((0, len(prompt.text)))
            await pilot.pause()

            assert app.completion_visible
            assert "main.py" in app.completion_values
            await pilot.press("enter")
            assert prompt.text == "检查 @{main.py}"

            await pilot.press("ctrl+enter")
            for _ in range(30):
                await pilot.pause(0.02)
                if model.requests and not app.busy:
                    break

            sent = model.requests[0][-1]["content"]
            assert '<file path="main.py" truncated="false">' in sent
            assert "print('引用内容')" in sent

    asyncio.run(exercise())


def test_at_completion_can_navigate_directories(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    (nested / "main.py").write_text("pass", encoding="utf-8")

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(30):
                await pilot.pause(0.02)
                if "src/package/main.py" in app.file_paths:
                    break
            prompt = app.query_one(PromptArea)
            prompt.text = "检查 @sr"
            prompt.move_cursor((0, len(prompt.text)))
            await pilot.pause()

            assert "src/" in app.completion_values
            app._insert_completion(app.completion_values.index("src/"))
            await pilot.pause()

            assert prompt.text == "检查 @{src/"
            assert "src/package/" in app.completion_values
            assert "src/package/main.py" in app.completion_values

    asyncio.run(exercise())


def test_tool_card_is_collapsed_with_key_argument_and_bounded_summary(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("工具摘要", encoding="utf-8")

    class ToolModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return AssistantTurn(
                    None,
                    (
                        ToolCall(
                            "read",
                            "read_file",
                            '{"path":"README.md","start_line":1,"end_line":10}',
                        ),
                    ),
                )
            return AssistantTurn("完成")

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), ToolModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "读取文件"
            await pilot.press("ctrl+enter")
            for _ in range(30):
                await pilot.pause(0.02)
                if not app.busy and "read" in app.tool_cards:
                    break

            card = app.tool_cards["read"]
            body = app.tool_bodies["read"]
            assert isinstance(card, Collapsible)
            assert card.collapsed
            assert card.title == "✓ read_file · README.md · 1–10"
            assert "状态：成功" in str(body.render())
            assert "工具摘要" in str(body.render())
            assert '"path"' not in str(body.render())

    asyncio.run(exercise())


def test_tui_renders_first_stream_delta_before_completion(tmp_path: Path) -> None:
    first_delta = threading.Event()
    release = threading.Event()

    class StreamingModel(FakeModel):
        def stream(self, messages, tools):
            self.requests.append(list(messages))
            yield ModelDelta(content="流")
            first_delta.set()
            release.wait(timeout=2)
            yield ModelDelta(content="式", finish_reason="stop")

    async def exercise() -> None:
        model = StreamingModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "测试流式"
            await pilot.press("ctrl+enter")
            for _ in range(30):
                await pilot.pause(0.02)
                if first_delta.is_set() and app.streaming_buffer == "流":
                    break

            assert app.busy
            assert app.streaming_buffer == "流"
            assert app.streaming_markdown is not None

            release.set()
            for _ in range(30):
                await pilot.pause(0.02)
                if not app.busy:
                    break

            assert not app.busy
            assert len(list(app.query(Markdown))) == 1

    try:
        asyncio.run(exercise())
    finally:
        release.set()
