from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Markdown, Static

from litcode_agent.config import Settings
from litcode_agent.model import AssistantTurn, ToolCall
from litcode_agent.tui import ConfirmCommand, LitCodeTUI, ModelPicker, PromptArea


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
