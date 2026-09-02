from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
from rich.style import Style
from textual.widgets import Collapsible, Input, Label, Markdown, OptionList, Static, Tree
from textual.events import MouseScrollDown

from litcode_agent.agent import AgentEvent
from litcode_agent.config import Settings
from litcode_agent.model import AssistantTurn, ModelDelta, ToolCall
from litcode_agent.tui import (
    COMMANDS,
    ChoicePicker,
    ConfirmCommand,
    HistoryPicker,
    LitCodeTUI,
    ModelPicker,
    PaneDivider,
    PromptArea,
    SkillPicker,
    WelcomeBanner,
    run_tui,
)


async def wait_until(pilot, predicate, attempts: int = 200, delay: float = 0.05):
    """Poll a TUI condition until it holds; CI 平台越慢，单次 pause 越不可靠。"""
    for _ in range(attempts):
        if predicate():
            return True
        await pilot.pause(delay)
    return predicate()


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
        {
            "HOME": str(tmp_path / "home"),
            "OPENAI_API_KEY": "secret",
            "LITCODE_MODEL": "model-a",
        },
    )


def test_tui_mounts_status_timeline_and_fixed_prompt(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)):
            header = str(app.query_one(".pane-header", Static).render())
            assert header.startswith("1 ")
            assert "空窗格" in header
            assert app.store.list_sessions(tmp_path) == ()
            assert len(app.query("#status")) == 0
            assert app.query_one(PromptArea).has_focus
            banner = app.query_one(WelcomeBanner)
            rendered = str(banner.render())
            assert "/help" in rendered
            assert "Ctrl+W" in rendered
            meta = str(app.query_one("#prompt-meta-left", Label).render())
            assert "● 就绪" in meta
            assert "model-a" in meta
            assert "environment" in meta
            meta_right = str(app.query_one("#prompt-meta-right", Label).render())
            assert "工作区" in meta_right
            assert tmp_path.name in meta_right

    asyncio.run(exercise())


def test_skill_command_opens_searchable_picker_and_inserts_invocation(
    tmp_path: Path,
) -> None:
    skill = tmp_path / ".agents" / "skills" / "review-code"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review-code\ndescription: Review changes carefully.\n---\nbody",
        encoding="utf-8",
    )

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_command("/skill")
            await pilot.pause()

            assert isinstance(app.screen, SkillPicker)
            skill_list = app.screen.query_one("#skill-list", OptionList)
            assert "Review changes carefully" in str(
                skill_list.get_option_at_index(0).prompt
            )
            await pilot.press("r", "e", "v", "i", "e", "w", "enter")
            await pilot.pause()

            assert app.query_one(PromptArea).text == "$review-code "
            notices = [
                str(widget.render()) for widget in app.query(".notice").results(Static)
            ]
            assert not any("Review changes carefully" in item for item in notices)

    asyncio.run(exercise())


def test_welcome_banner_disappears_after_first_message(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.query_one(WelcomeBanner) is not None
            prompt = app.query_one(PromptArea)
            prompt.text = "你好"
            prompt.action_submit()
            assert await wait_until(
                pilot, lambda: not list(app.query(WelcomeBanner))
            )
            assert not list(app.query(WelcomeBanner))

    asyncio.run(exercise())


def test_escape_interrupts_running_reply(tmp_path: Path) -> None:
    import time

    class SlowModel(FakeModel):
        def complete(self, messages, tools):
            time.sleep(0.3)
            return AssistantTurn("回答")

    async def exercise() -> None:
        model = SlowModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "开始"
            await pilot.press("enter")
            await pilot.pause()
            assert app.busy

            await pilot.press("escape")
            await pilot.pause()

            runtime = app._active_runtime()
            assert runtime.cancel_requested.is_set()
            assert "中断" in str(
                app.query_one("#prompt-status-left", Label).render()
            )
            for _ in range(160):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            assert not app.busy
            assert list(app.query("#prompt-status-left"))[0].render() == ""

    asyncio.run(exercise())


def test_closed_pane_releases_its_number_for_the_next_split(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(160, 44)) as pilot:
            app.action_split("right")
            app.action_split("down")
            await pilot.pause()

            assert app.active_pane_id == "pane-3"
            app.action_close_pane()
            await pilot.pause()
            app.action_split("left")
            await pilot.pause()

            assert app.active_pane_id == "pane-3"
            headers = [
                str(widget.render())
                for widget in app.query(".pane-header").results(Static)
            ]
            assert {header[0] for header in headers} == {"1", "2", "3"}

    asyncio.run(exercise())


def test_divider_drag_updates_layout_ratio_and_uses_a_real_empty_pane(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(160, 44)) as pilot:
            app.action_split("right")
            await pilot.pause()
            assert app.store.list_sessions(tmp_path) == ()
            divider = app.query_one(PaneDivider)
            assert divider.axis == "horizontal"
            app.resize_pane(divider.target_pane_id, divider.axis, 0.2)
            await pilot.pause()
            root = app.pane_layout.root
            assert root.ratio == pytest.approx(0.7)  # type: ignore[union-attr]
            assert app.query(".pane-divider")

    asyncio.run(exercise())


def test_queue_command_can_cancel_and_reorder_pending_user_messages(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            session_id = app.session.session_id
            app.runtime.pause(session_id)
            first = app.store.enqueue_message(session_id, "first")
            second = app.store.enqueue_message(session_id, "second")
            app.action_queue(f"up {second.id[:8]}")
            assert [item.content for item in app.store.queue(session_id)] == [
                "second",
                "first",
            ]
            app.action_queue(f"cancel {first.id[:8]}")
            assert [item.content for item in app.store.queue(session_id)] == ["second"]
            await pilot.pause()

    asyncio.run(exercise())


def test_model_request_receives_current_terminal_and_pane_location(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            app.action_split("right")
            await pilot.pause()
            assert isinstance(app.screen, ChoicePicker)
            await pilot.press("enter")
            await pilot.pause()
            prompt = app.query_one(PromptArea)
            prompt.text = "我在哪里"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break

            runtime_messages = [
                str(message["content"])
                for message in model.requests[0]
                if message["role"] == "system"
            ]
            assert any("当前 pane：2" in content for content in runtime_messages)
            assert any(
                app.sessions.terminal_id in content for content in runtime_messages
            )
            assert all(
                "litcode_runtime_location" not in str(message["content"])
                for message in app.session.messages
            )

    asyncio.run(exercise())


def test_schedule_command_sends_natural_language_as_explicit_tool_request(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            tool_names = {
                schema["function"]["name"] for schema in app.registry.schemas()
            }
            assert {
                "create_scheduled_task",
                "list_scheduled_tasks",
                "cancel_scheduled_task",
            } <= tool_names
            app._handle_command("/schedule 每周一上午九点运行测试")
            for _ in range(120):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break

            request = str(model.requests[0][-1]["content"])
            assert "create_scheduled_task" in request
            assert "每周一上午九点运行测试" in request
            assert "IANA 时区" in request

    asyncio.run(exercise())


def test_session_picker_focuses_an_already_mounted_pane(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            app.action_split("right")
            await pilot.pause()
            pane_two_session = app.session.session_id
            app.action_focus_pane("left")

            app._session_selected(pane_two_session)

            assert app.active_pane_id == "pane-2"
            assert app.session.session_id == pane_two_session
            assert len(
                [
                    pane
                    for pane in app.sessions.panes.values()
                    if pane.session is not None
                    and pane.session.session_id == pane_two_session
                ]
            ) == 1

    asyncio.run(exercise())


def test_hash_completion_lists_mounted_panes_in_numeric_order(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            first = app.session.session_id
            app.action_split("right")
            await pilot.pause()
            second = app.session.session_id
            prompt = app.query_one(PromptArea)
            prompt.text = "#"
            prompt.move_cursor(prompt.document.end)
            app.refresh_completions()
            await pilot.pause()

            assert app.completion_values[:2] == [
                app.store.session_info(first).alias,
                app.store.session_info(second).alias,
            ]

    asyncio.run(exercise())


def test_enter_sends_and_shift_enter_inserts_newline(tmp_path: Path) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "第一行"
            prompt.move_cursor(prompt.document.end)
            await pilot.press("shift+enter")
            await pilot.pause()

            assert prompt.text == "第一行\n"
            assert model.requests == []

            prompt.text += "第二行"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break

            assert model.requests[0][-1]["content"] == "第一行\n第二行"

    asyncio.run(exercise())


def test_prompt_history_recalls_messages_and_restores_draft(tmp_path: Path) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "第一条"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break

            prompt.text = "未发送草稿"
            prompt.move_cursor((0, 0))
            await pilot.press("up")
            assert prompt.text == "第一条"

            await pilot.press("down")
            assert prompt.text == "未发送草稿"

            prompt.text = "第一行\n中间行\n最后行"
            prompt.move_cursor((1, 0))
            await pilot.press("up")
            assert prompt.text == "第一行\n中间行\n最后行"
            assert prompt.cursor_location == (0, 0)

    asyncio.run(exercise())


def test_ctrl_c_requires_second_press_to_exit(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+c")
            assert not app._exit

            await pilot.press("ctrl+c")
            assert app._exit

    asyncio.run(exercise())


def test_exit_command_exits_without_second_press(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_command("/exit")
            await pilot.pause()
            assert app._exit

    asyncio.run(exercise())


def test_quit_alias_exits_without_second_press(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_command("/quit")
            await pilot.pause()
            assert app._exit

    asyncio.run(exercise())


def test_run_tui_enables_mouse_for_wheel_and_scrollbar(
    tmp_path: Path, monkeypatch
) -> None:
    options: dict[str, object] = {}

    def fake_run(app: LitCodeTUI, **kwargs: object) -> None:
        options.update(kwargs)
        app.store.close()

    monkeypatch.setattr(LitCodeTUI, "run", fake_run)

    assert run_tui(settings(tmp_path), FakeModel()) == 0  # type: ignore[arg-type]
    assert options == {"mouse": True}


def test_mouse_wheel_scrolls_each_split_timeline_independently(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 24)) as pilot:
            original = app._active_runtime()
            snapshot_count = len(app._timeline(original).children)
            app.action_split("right")
            assert await wait_until(
                pilot,
                lambda: len(app._timeline(original).children) == snapshot_count
                and len(app.query("#timeline-pane-2")) > 0,
            )
            for runtime in app.panes.values():
                for number in range(40):
                    app._append_notice(f"line {number}", runtime=runtime)
            await pilot.pause()

            for runtime in app.panes.values():
                timeline = app._timeline(runtime)
                assert timeline.max_scroll_y > 0
                timeline.scroll_home(animate=False)
                await pilot.pause()
                before = timeline.scroll_y
                timeline.post_message(
                    MouseScrollDown(
                        timeline, 1, 1, 0, 1, 0, False, False, False
                    )
                )
                await pilot.pause()
                assert timeline.scroll_y > before

    asyncio.run(exercise())


def test_split_preserves_original_pane_visual_timeline(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 30)) as pilot:
            original = app._active_runtime()
            app._append_notice("SPLIT-PRESERVE-MARKER", runtime=original)
            await pilot.pause()

            app.action_split("right")
            assert await wait_until(
                pilot,
                lambda: any(
                    "SPLIT-PRESERVE-MARKER" in item
                    for item in [
                        str(widget.render())
                        for widget in app.query(
                            "#view-pane-1 .notice"
                        ).results(Static)
                    ]
                ),
            )

            original_notices = [
                str(widget.render())
                for widget in app.query("#view-pane-1 .notice").results(Static)
            ]
            assert any("SPLIT-PRESERVE-MARKER" in item for item in original_notices)

    asyncio.run(exercise())


def test_split_opens_new_pane_session_choice_and_cancel_rolls_back(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        app.store.create(tmp_path, "model-a", [], title="可挂载历史")
        async with app.run_test(size=(120, 30)) as pilot:
            original = app.session.session_id

            app.action_split("right")
            await pilot.pause()

            assert isinstance(app.screen, ChoicePicker)
            labels = [label for _, label in app.screen.choices]
            assert any("新会话" in label for label in labels)
            assert any("可挂载历史" in label for label in labels)
            app.screen.action_cancel()
            await pilot.pause()

            assert tuple(app.panes) == ("pane-1",)
            assert app.session.session_id == original

    asyncio.run(exercise())


def test_split_choice_keeps_new_session_as_draft_or_mounts_history(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        target = app.store.create(tmp_path, "model-a", [], title="待挂载历史")
        async with app.run_test(size=(120, 30)) as pilot:
            original = app.session.session_id

            app.action_split("right")
            await pilot.pause()
            assert isinstance(app.screen, ChoicePicker)
            await pilot.press("enter")
            await pilot.pause()
            assert app.active_pane_id == "pane-2"
            assert app.panes["pane-2"].session is None
            assert app.panes["pane-1"].session.session_id == original

            app.action_close_pane()
            await pilot.pause()
            app.action_split("right")
            await pilot.pause()
            assert isinstance(app.screen, ChoicePicker)
            app.screen.dismiss(target)
            await pilot.pause()
            assert app.active_pane_id == "pane-2"
            assert app.session.session_id == target
            assert app.panes["pane-1"].session.session_id == original

    asyncio.run(exercise())


def test_tui_explains_portable_split_shortcut_on_start(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)):
            notices = [str(widget.render()) for widget in app.query(".notice").results(Static)]

            assert any("Ctrl+W 后按方向键" in notice for notice in notices)
            assert any("/split right" in notice for notice in notices)

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
            await pilot.press("enter")
            for _ in range(80):
                await pilot.pause(0.05)
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


def test_assistant_markdown_has_clear_semantic_contrast(tmp_path: Path) -> None:
    class MarkdownModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            return AssistantTurn(
                "# 结论\n\n正文里的 **重点** 和 `value`。\n\n"
                "> 注意事项\n\n```python\nprint('ok')\n```"
            )

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), MarkdownModel())  # type: ignore[arg-type]
        async with app.run_test(size=(100, 36)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "展示 Markdown"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy:
                    break

            heading = app.query_one("MarkdownH1")
            fence = app.query_one("MarkdownFence")
            quote = app.query_one("MarkdownBlockQuote")
            paragraph = app.query_one("MarkdownParagraph")

            assert heading.styles.text_style.bold
            assert heading.styles.border_bottom[0] == "solid"
            assert fence.styles.border_left[0] == "round"
            assert quote.styles.border_left[0] == "thick"

            strong = paragraph.get_component_rich_style("strong")
            inline_code = paragraph.get_component_rich_style("code_inline")
            assert strong.bold
            assert strong.color != paragraph.rich_style.color
            assert inline_code.bold
            assert inline_code.bgcolor is not None

    asyncio.run(exercise())


def test_tui_queues_messages_and_accepts_commands_while_busy(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            session_id = app.session.session_id
            app.runtime.pause(session_id)

            prompt = app.query_one(PromptArea)
            prompt.text = "第一条"
            await pilot.press("enter")
            assert app.busy
            assert not prompt.disabled

            prompt.text = "第二条"
            await pilot.press("enter")
            await pilot.pause()
            queue = app.store.queue(app.session.session_id)
            assert [item.content for item in queue] == ["第一条", "第二条"]
            strip = app.query_one("#prompt-queue", Static)
            assert "⏳" in str(strip.render())
            assert "第二条" in str(strip.render())
            user_bundles = [
                str(widget.render())
                for widget in app.query(".message-user").results(Static)
            ]
            assert len(user_bundles) == 1
            assert "第一条" in user_bundles[0]
            assert "第二条" not in " ".join(user_bundles)

            prompt.text = "/tree"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, HistoryPicker)
            app.screen.action_cancel()
            await pilot.pause()

            app.runtime.resume(session_id)
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy and len(model.requests) == 2:
                    break
            assert len(model.requests) == 2
            assert not app.busy

    asyncio.run(exercise())


def test_tui_model_picker_switches_current_model(tmp_path: Path) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("f2")
            for _ in range(80):
                await pilot.pause(0.05)
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
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if isinstance(app.screen, ConfirmCommand):
                    break
            assert isinstance(app.screen, ConfirmCommand)

            await pilot.press("escape")
            for _ in range(120):
                await pilot.pause(0.05)
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

            prompt.text = "/se"
            prompt.move_cursor((0, 3))
            await pilot.pause()
            assert "/history" in app.completion_values
            assert "/sessions" not in app.completion_values

            prompt.text = "/t"
            prompt.move_cursor((0, 2))
            await pilot.pause()
            assert "/history" in app.completion_values
            assert "/tree" not in app.completion_values

    asyncio.run(exercise())


def test_command_registry_includes_session_workflows() -> None:
    assert {item.name for item in COMMANDS} >= {
        "/history",
        "/compact",
        "/rewind",
        "/redo",
        "/fork",
    }
    history = next(item for item in COMMANDS if item.name == "/history")
    assert {"/sessions", "/tree", "/resume"} <= set(history.aliases)


def test_new_command_detaches_session_and_keeps_running_task(
    tmp_path: Path,
) -> None:
    class BlockingModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()

        def complete(self, messages, tools):
            self.requests.append(list(messages))
            self.release.wait(timeout=5)
            return AssistantTurn("TUI 回答")

    async def exercise() -> None:
        model = BlockingModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "问题"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if app.busy:
                    break
            previous = app._active_runtime().session
            assert app.busy

            app._handle_command("/new")
            assert await wait_until(pilot, lambda: not app.busy)

            assert app.active_pane_id == "pane-1"
            assert app.session.session_id != previous.session_id
            assert not previous.closed
            assert previous.session_id in app.sessions.detached
            assert previous.session_id in app.running_sessions
            assert not app.busy
            assert await wait_until(
                pilot,
                lambda: not any(
                    "问题" in str(child.render())
                    for child in app._timeline(app._active_runtime()).children
                ),
            )
            timeline = app._timeline(app._active_runtime())

            app._session_selected(previous.session_id)
            assert await wait_until(
                pilot, lambda: app.session.session_id == previous.session_id
            )
            assert app.session.session_id == previous.session_id
            assert app.busy
            assert not app.query_one(PromptArea).disabled
            assert previous.session_id in app.running_sessions

            model.release.set()
            for _ in range(200):
                await pilot.pause(0.05)
                if previous.session_id not in app.running_sessions:
                    break
            assert previous.session_id not in app.sessions.detached
            assert any(
                message.get("content") == "TUI 回答"
                for message in previous.messages
            )
            assert app.busy is False
            assert not app.query_one(PromptArea).disabled
            assert app.session.messages[-1]["content"] == "TUI 回答"

    asyncio.run(exercise())


def test_nohup_returns_last_pane_to_empty_without_creating_a_session(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            previous = app.session.session_id
            before = {item.id for item in app.store.list_sessions(tmp_path)}
            app._handle_command("/nohup")
            await pilot.pause()

            assert app._active_runtime().empty
            assert {item.id for item in app.store.list_sessions(tmp_path)} == before
            assert "空窗格" in str(app.query_one(".pane-header", Static).render())

            prompt = app.query_one(PromptArea)
            prompt.text = "新的根会话"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break
            assert app.session.session_id != previous
            assert len(app.store.list_sessions(tmp_path)) == len(before) + 1

    asyncio.run(exercise())


def test_split_can_mount_an_existing_background_session_without_a_ghost(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        target = app.store.create(tmp_path, "model-a", [], title="后台调查")
        alias = app.store.session_info(target).alias
        async with app.run_test(size=(120, 40)) as pilot:
            before = {item.id for item in app.store.list_sessions(tmp_path)}
            app._handle_command(f"/split right {alias}")
            await pilot.pause()

            assert app.active_pane_id == "pane-2"
            assert app.session.session_id == target
            assert {item.id for item in app.store.list_sessions(tmp_path)} == before
            assert len(
                [
                    pane
                    for pane in app.sessions.panes.values()
                    if pane.session is not None and pane.session.session_id == target
                ]
            ) == 1

    asyncio.run(exercise())


def test_subagent_command_can_create_mount_and_start_a_child_pane(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            parent_id = app.session.session_id

            app._handle_command("/subagent --pane right 检查测试")
            for _ in range(160):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break

            assert app.active_pane_id == "pane-2"
            child = app.store.session_info(app.session.session_id)
            assert child.parent_id == parent_id
            assert model.requests[0][-1]["content"] == "检查测试"
            assert app.session.messages[-1]["content"] == "TUI 回答"
            assert await wait_until(
                pilot,
                lambda: any(
                    "挂载到 2 号 pane" in str(widget.render())
                    for widget in app.query(".notice").results(Static)
                ),
            )
            assert any(
                "挂载到 2 号 pane" in str(widget.render())
                for widget in app.query(".notice").results(Static)
            )

    asyncio.run(exercise())


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
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            app._handle_command("/compact")
            for _ in range(120):
                await pilot.pause(0.05)
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
            for _ in range(120):
                await pilot.pause(0.05)
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

            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
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
            for _ in range(120):
                await pilot.pause(0.05)
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


def test_hash_completion_inserts_session_alias_and_sends_capsule(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        source = app.store.create(
            tmp_path,
            "model-a",
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "检查缓存"},
                {"role": "assistant", "content": "cache key 缺少 workspace"},
            ],
            title="缓存调查",
        )
        alias = app.store.session_info(source).alias
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = f"参考 #{alias}"
            prompt.move_cursor((0, len(prompt.text)))
            await pilot.pause()

            assert alias in app.completion_values
            await pilot.press("enter")
            assert prompt.text == f"参考 #{{{alias}}}"

            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break

            sent = model.requests[0][-1]["content"]
            assert f'<session_reference alias="{alias}"' in sent
            assert "缓存调查" in sent
            assert "cache key 缺少 workspace" in sent

    asyncio.run(exercise())


def test_tui_advertises_skill_metadata_without_eager_body(tmp_path: Path) -> None:
    skill = tmp_path / ".agents" / "skills" / "review-code"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review-code\ndescription: Use when reviewing code.\n---\n"
        "PRIVATE SKILL BODY",
        encoding="utf-8",
    )

    async def exercise() -> None:
        model = FakeModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "检查代码"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if model.requests and not app.busy:
                    break

            system = model.requests[0][0]["content"]
            assert "review-code" in system
            assert "Use when reviewing code." in system
            assert "PRIVATE SKILL BODY" not in system

    asyncio.run(exercise())


def test_split_panes_run_concurrently_and_keep_streams_isolated(
    tmp_path: Path,
) -> None:
    slow_started = threading.Event()
    release_slow = threading.Event()

    class ConcurrentModel(FakeModel):
        def stream(self, messages, tools):
            self.requests.append(list(messages))
            task = messages[-1]["content"]
            if task == "慢任务":
                yield ModelDelta(content="慢")
                slow_started.set()
                release_slow.wait(timeout=2)
                yield ModelDelta(content="完成", finish_reason="stop")
            else:
                yield ModelDelta(content="快完成", finish_reason="stop")

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), ConcurrentModel())  # type: ignore[arg-type]
        async with app.run_test(size=(160, 44)) as pilot:
            app._handle_command("/split right")
            await pilot.pause()
            assert len(app.panes) == 2
            assert app.active_pane_id == "pane-2"
            assert isinstance(app.screen, ChoicePicker)
            await pilot.press("enter")
            await pilot.pause()

            app._handle_command("/focus left")
            prompt = app.query_one(PromptArea)
            prompt.text = "慢任务"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if slow_started.is_set():
                    break

            assert app.panes["pane-1"].busy
            app._handle_command("/focus right")
            assert not prompt.disabled
            prompt.text = "快任务"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.panes["pane-2"].busy:
                    break

            assert app.panes["pane-1"].streaming_buffer == "慢"
            fast_timeline = app.query_one("#timeline-pane-2")
            assert len(list(fast_timeline.query(Markdown))) == 1
            assert app.panes["pane-2"].session.messages[-1]["content"] == "快完成"

            release_slow.set()
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.panes["pane-1"].busy:
                    break
            assert not app.panes["pane-1"].busy

    try:
        asyncio.run(exercise())
    finally:
        release_slow.set()


def test_ctrl_w_direction_is_portable_split_fallback(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.press("ctrl+w", "right")
            await pilot.pause()

            assert len(app.panes) == 2
            assert app.active_pane_id == "pane-2"

            await pilot.press("ctrl+w", "h")
            assert app.active_pane_id == "pane-1"

            await pilot.press("ctrl+w", "w")
            assert app.active_pane_id == "pane-2"

    asyncio.run(exercise())


def test_repeated_ctrl_w_cycles_panes_until_sequence_stops(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(160, 44)) as pilot:
            app.action_split("right")
            app.action_split("down")
            await pilot.pause()
            app.action_focus_pane("left")
            assert app.active_pane_id == "pane-1"

            await pilot.press("ctrl+w", "ctrl+w", "ctrl+w")

            assert app.active_pane_id == "pane-3"
            assert app.pane_leader_active

            await pilot.pause(1.3)
            assert not app.pane_leader_active

    asyncio.run(exercise())


def test_expired_ctrl_w_prefix_does_not_consume_a_late_key(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)):
            app.action_pane_leader()
            app._pane_leader_until = 0.0

            assert not app.consume_pane_leader_key("right")
            assert len(app.panes) == 1

    asyncio.run(exercise())


def test_closing_pane_detaches_without_ending_session(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            app.action_split("right")
            await pilot.pause()
            detached_id = app.session.session_id
            detached_session = app.session

            app.action_close_pane()
            await pilot.pause()

            assert not detached_session.closed
            assert detached_id in app.sessions.detached
            app._session_selected(detached_id)
            await pilot.pause()
            assert app.session is detached_session
            assert not app.session.closed

    asyncio.run(exercise())


def test_event_for_unmounted_session_is_not_routed_to_active_pane(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        rendered: list[str] = []
        async with app.run_test(size=(120, 40)):
            app._render_agent_event = (  # type: ignore[method-assign]
                lambda event, runtime: rendered.append(runtime.pane_id)
            )
            app.receive_agent_event(
                AgentEvent(
                    kind="model_delta",
                    iteration=1,
                    content="late",
                    session_id="gone",
                )
            )

            assert rendered == []

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
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy and "read" in app._active_runtime().tool_cards:
                    break

            card = app._active_runtime().tool_cards["read"]
            body = app._active_runtime().tool_bodies["read"]
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
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if (
                    first_delta.is_set()
                    and app._active_runtime().streaming_buffer == "流"
                ):
                    break

            assert app.busy
            assert app._active_runtime().streaming_buffer == "流"
            assert app._active_runtime().streaming_markdown is not None

            release.set()
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy:
                    break

            assert not app.busy
            assert len(list(app.query(Markdown))) == 1

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_apply_patch_card_shows_bounded_diff_for_creation(tmp_path: Path) -> None:
    from litcode_agent.tools.base import FileChange

    class ToolModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return AssistantTurn(None, (ToolCall("create", "apply_patch", '{"path":"new.py","old_text":"","new_text":"one\\ntwo\\nthree"}'),))
            return AssistantTurn("完成")

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), ToolModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            event_body = app._active_runtime().tool_bodies
            prompt = app.query_one(PromptArea)
            prompt.text = "创建文件"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy and "create" in event_body:
                    break

            body = event_body["create"].render()
            text = str(body)
            assert "+one" in text
            assert "+two" in text
            assert "+three" in text
            assert "@@ -0,0 +1,3 @@" in text
            assert '"path"' not in text

    asyncio.run(exercise())


def test_paged_output_bounds_pages_and_navigates(tmp_path: Path) -> None:
    from litcode_agent.tui import PagedOutput

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        runtime = app._active_runtime()
        async with app.run_test(size=(120, 40)):
            output = PagedOutput("line\n" * 30)
            assert output.page_count == 2
            page_one = str(output.render())
            assert "第 1/2 页" in page_one
            output.action_next_page()
            assert "第 2/2 页" in str(output.render())
            output.action_next_page()
            assert "第 2/2 页" in str(output.render())
            output.action_prev_page()
            assert "第 1/2 页" in str(output.render())

    asyncio.run(exercise())


def test_history_picker_filters_and_selected_session_switches_pane(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(140, 40)) as pilot:
            first = app.session.session_id
            (tmp_path / "placeholder").write_text("x", encoding="utf-8")
            other = app.store.create(tmp_path, "model-a", [], title="其他会话")
            app._handle_command("/history")
            await pilot.pause()
            await pilot.pause()

            picker = app.screen
            assert isinstance(picker, HistoryPicker)
            help_text = str(picker.query_one("#history-help", Label).render())
            assert "共 2 个会话" in help_text
            assert _tree_node_datas(picker) == [other, first]
            tree = picker.query_one("#history-tree", Tree)
            assert tree.cursor_node is not None
            assert tree.cursor_node.data == other
            assert "当前" in tree.root.children[1].label.plain
            cursor_style = tree.get_component_rich_style("tree--cursor")
            assert cursor_style.bold
            assert cursor_style.bgcolor is not None

            filter_input = picker.query_one("#history-filter", Input)
            filter_input.value = "其他"
            await pilot.pause()
            assert _tree_node_datas(picker) == [other]
            assert "当前" not in picker.query_one("#history-tree", Tree).root.children[0].label.plain

            app._session_selected(other)
            await pilot.pause()
            assert app.session.session_id == other
            assert first in app.sessions.detached

    asyncio.run(exercise())


def test_history_tree_uses_left_right_for_hierarchy_navigation(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 36)) as pilot:
            parent_id = app.session.session_id
            child_id = app.store.create_child(parent_id, title="子调查")
            app._handle_command("/history")
            await pilot.pause()
            await pilot.pause()

            picker = app.screen
            assert isinstance(picker, HistoryPicker)
            tree = picker.query_one("#history-tree", Tree)
            assert tree.has_focus
            parent = tree.cursor_node
            assert parent is not None and parent.data == parent_id
            assert not parent.is_expanded

            await pilot.press("right")
            assert parent.is_expanded
            await pilot.press("right")
            assert tree.cursor_node is not None
            assert tree.cursor_node.data == child_id
            await pilot.press("left")
            assert tree.cursor_node is parent
            await pilot.press("left")
            assert not parent.is_expanded

    asyncio.run(exercise())


def test_history_tree_keeps_right_timestamp_inside_narrow_viewport(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        app.store.create(tmp_path, "model-a", [], title="一个很长的历史会话标题" * 4)
        async with app.run_test(size=(72, 24)) as pilot:
            app._handle_command("/history")
            await pilot.pause()
            await pilot.pause()

            picker = app.screen
            assert isinstance(picker, HistoryPicker)
            tree = picker.query_one("#history-tree", Tree)
            node = tree.cursor_node
            assert node is not None
            rendered = tree.render_label(node, Style(), Style())

            assert rendered.plain.rstrip().endswith("刚刚")
            assert rendered.cell_len <= tree.scrollable_content_region.width

    asyncio.run(exercise())


def test_history_shift_enter_mounts_selection_in_a_new_pane(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        other = app.store.create(tmp_path, "model-a", [], title="右侧会话")
        async with app.run_test(size=(140, 40)) as pilot:
            app._handle_command("/history")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, HistoryPicker)

            await pilot.press("shift+enter")
            await pilot.pause()

            assert app.active_pane_id == "pane-2"
            assert app.session.session_id == other

    asyncio.run(exercise())


def test_history_switch_keeps_running_task_in_background(tmp_path: Path) -> None:
    blocked = threading.Event()

    class BlockingModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            blocked.wait(timeout=10)
            return AssistantTurn("后台回答")

    async def exercise() -> None:
        model = BlockingModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            busy_id = app.session.session_id
            prompt = app.query_one(PromptArea)
            prompt.text = "慢任务"
            await pilot.press("enter")
            for _ in range(240):
                await pilot.pause(0.05)
                if app.busy:
                    break
            assert app.busy
            other = app.store.create(tmp_path, "model-a", [], title="切换目标")

            app._session_selected(other)
            assert await wait_until(
                pilot, lambda: app.session.session_id == other and not app.busy
            )

            assert app.session.session_id == other
            assert busy_id in app.sessions.detached
            assert busy_id in app.running_sessions
            assert not app.busy
            assert not app.query_one(PromptArea).disabled

            blocked.set()
            for _ in range(200):
                await pilot.pause(0.05)
                if busy_id not in app.running_sessions:
                    break
            assert busy_id not in app.running_sessions
            assert any(
                message.get("content") == "后台回答"
                for message in app.sessions.detached[busy_id].messages
            )

    try:
        asyncio.run(exercise())
    finally:
        blocked.set()


def test_tui_ask_user_modal_returns_answer_and_cards(tmp_path: Path) -> None:
    from litcode_agent.tui import QuestionPrompt

    class ToolModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return AssistantTurn(
                    None,
                    (
                        ToolCall(
                            "q1",
                            "ask_user",
                            '{"questions":[{"header":"方向","question":"继续还是回退？",'
                            '"options":[{"label":"继续","description":"按计划继续"},'
                            '{"label":"回退","description":"撤销前一步"}]}]}',
                        ),
                    ),
                )
            return AssistantTurn("完成")

    async def exercise() -> None:
        model = ToolModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "选择方向"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if isinstance(app.screen, QuestionPrompt):
                    break
            assert isinstance(app.screen, QuestionPrompt)
            assert await wait_until(
                pilot,
                lambda: "继续还是回退？"
                in str(app.screen.query_one("#question-text", Static).render()),
            )
            assert "继续还是回退？" in str(app.screen.query_one("#question-text", Static).render())

            await pilot.press("1")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy and len(model.requests) == 2:
                    break

            tool_message = model.requests[1][-1]
            payload = json.loads(tool_message["content"])
            assert payload["ok"] is True
            assert (
                'User has answered your questions: "继续还是回退？"="继续"'
                in payload["content"]
            )
            card = app._active_runtime().tool_cards["q1"]
            assert card.title == "✓ ask_user · 1 个问题"
            body = str(app._active_runtime().tool_bodies["q1"].render())
            assert "继续" in body

    asyncio.run(exercise())


def test_tui_ask_user_escape_returns_tool_error(tmp_path: Path) -> None:
    from litcode_agent.tui import QuestionPrompt

    class ToolModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                return AssistantTurn(
                    None,
                    (
                        ToolCall(
                            "q1",
                            "ask_user",
                            '{"questions":[{"header":"方向","question":"继续还是回退？",'
                            '"options":[{"label":"继续","description":"a"},{"label":"回退","description":"b"}]}]}',
                        ),
                    ),
                )
            return AssistantTurn("继续执行")

    async def exercise() -> None:
        model = ToolModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "选择方向"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if isinstance(app.screen, QuestionPrompt):
                    break
            assert isinstance(app.screen, QuestionPrompt)

            await pilot.press("escape")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy and len(model.requests) == 2:
                    break

            tool_message = model.requests[1][-1]
            payload = json.loads(tool_message["content"])
            assert payload["ok"] is False
            assert "用户取消了提问" in payload["content"]
            assert app._active_runtime().tool_cards["q1"].title.startswith("✗")

    asyncio.run(exercise())


def test_tui_ask_user_multi_question_confirm_tab(tmp_path: Path) -> None:
    import json as json_module

    from litcode_agent.tui import QuestionPrompt

    class ToolModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                questions = [
                    {
                        "header": "方向",
                        "question": "继续还是回退？",
                        "options": [
                            {"label": "继续", "description": "a"},
                            {"label": "回退", "description": "b"},
                        ],
                    },
                    {
                        "header": "风格",
                        "question": "选择代码风格？",
                        "options": [
                            {"label": "简洁", "description": "c"},
                            {"label": "注释详尽", "description": "d"},
                        ],
                    },
                ]
                return AssistantTurn(
                    None,
                    (ToolCall("q1", "ask_user", json_module.dumps({"questions": questions})),),
                )
            return AssistantTurn("完成")

    async def exercise() -> None:
        model = ToolModel()
        app = LitCodeTUI(settings(tmp_path), model)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "多题"
            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if isinstance(app.screen, QuestionPrompt):
                    break
            assert isinstance(app.screen, QuestionPrompt)
            picker = app.screen

            await pilot.press("1")
            await pilot.pause()
            assert picker.tab == 1
            await pilot.press("2")
            await pilot.pause()
            assert picker.tab == picker.confirm_tab

            await pilot.press("enter")
            for _ in range(120):
                await pilot.pause(0.05)
                if not app.busy and len(model.requests) == 2:
                    break

            payload = json.loads(model.requests[1][-1]["content"])
            assert '"继续还是回退？"="继续"' in payload["content"]
            assert '"选择代码风格？"="注释详尽"' in payload["content"]

    asyncio.run(exercise())


def test_subagent_card_shows_spinner_model_and_completion_summary(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 36)) as pilot:
            runtime = app._active_runtime()
            call = ToolCall(
                "sub-1",
                "spawn_subagent",
                '{"prompt":"检查测试失败原因","agent":"explore"}',
            )
            app._append_tool(call, runtime)
            await pilot.pause(0.15)

            card = runtime.tool_cards[call.id]
            assert "子代理 [explore] · model-a" in str(card.title)
            assert str(card.title)[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            assert "任务：检查测试失败原因" in str(
                runtime.tool_bodies[call.id].render()
            )

            app._finish_tool(
                call,
                '{"invocation_id":"inv-1","session_id":"child-1",'
                '"alias":"260901-1432-ABC","background":false,'
                '"output":"失败来自 fixture 没有初始化。"}',
                False,
                runtime,
            )

            assert str(card.title).startswith("✓ 子代理 260901-1432-ABC")
            assert "model-a" in str(card.title)
            assert "完成摘要：失败来自 fixture 没有初始化。" in str(
                runtime.tool_bodies[call.id].render()
            )
            assert call.id not in runtime.subagent_cards

    asyncio.run(exercise())


def test_background_subagent_card_stays_live_until_child_finishes(
    tmp_path: Path,
) -> None:
    child_started = threading.Event()
    child_release = threading.Event()

    class BackgroundSubagentModel(FakeModel):
        def complete(self, messages, tools):
            self.requests.append(list(messages))
            last = messages[-1]
            content = last.get("content", "")
            if content == "启动后台":
                return AssistantTurn(
                    None,
                    (
                        ToolCall(
                            "sub-background",
                            "spawn_subagent",
                            '{"prompt":"检查测试","agent":"explore",'
                            '"background":true}',
                        ),
                    ),
                )
            if content == "检查测试":
                child_started.set()
                assert child_release.wait(15)
                return AssistantTurn("child 找到了失败原因")
            return AssistantTurn("父会话继续")

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), BackgroundSubagentModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 36)) as pilot:
            prompt = app.query_one(PromptArea)
            prompt.text = "启动后台"
            await pilot.press("enter")
            for _ in range(240):
                await pilot.pause(0.05)
                if child_started.is_set() and not app.busy:
                    break

            runtime = app._active_runtime()
            card = runtime.tool_cards["sub-background"]
            assert "子代理" in str(card.title)
            assert str(card.title)[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            assert "当前：" in str(
                runtime.tool_bodies["sub-background"].render()
            )

            child_release.set()
            for _ in range(240):
                await pilot.pause(0.05)
                if "sub-background" not in runtime.subagent_cards:
                    break

            assert str(card.title).startswith("✓ 子代理")
            assert "child 找到了失败原因" in str(
                runtime.tool_bodies["sub-background"].render()
            )

    asyncio.run(exercise())


def test_history_label_places_relative_time_on_the_right() -> None:
    import time as time_module

    from litcode_agent.session_store import SessionInfo
    from litcode_agent.tui import _cell_width, _history_label, _relative_time

    now = time_module.time()
    info = SessionInfo(
        id="s1",
        alias="260901-1400-ABC",
        title="测试任务",
        model="model-a",
        updated_at=now - 3 * 3600,
    )

    label = _history_label(info, {}, set(), width=60)
    assert "3h" in label.plain
    assert label.plain.rstrip().endswith("3h")
    assert "model-a" not in label.plain

    long_info = SessionInfo(
        id="s2",
        alias="260901-1400-ABC",
        title="很长的标题" * 20,
        model="model-a",
        updated_at=now - 3 * 3600,
    )
    long_label = _history_label(long_info, {}, set(), width=40)
    assert "…" in long_label.plain
    assert _cell_width(long_label.plain) <= 40

    assert _relative_time(now - 40) == "刚刚"
    assert _relative_time(now - 90) == "1m"
    assert _relative_time(now - 2 * 86400) == "2d"


def test_history_label_shows_child_model_and_current_activity() -> None:
    import time as time_module

    from litcode_agent.session_store import SessionInfo
    from litcode_agent.tui import _history_label

    info = SessionInfo(
        id="child",
        alias="260901-1400-CHILD",
        title="检查测试",
        model="model-child",
        updated_at=time_module.time(),
        parent_id="parent",
        status="running",
        activity="正在调用工具 · search_files",
        active_turn_id="turn-1",
    )

    label = _history_label(info, {}, {info.id}, width=100)

    assert "● 正在调用工具 · search_files" in label.plain
    assert "model-child" in label.plain


def _tree_node_datas(picker) -> list[str | None]:
    tree = picker.query_one("#history-tree", Tree)
    result: list[str | None] = []
    visited: set[int] = set()

    def visit(node) -> None:
        if node.label.plain != "会话":
            result.append(node.data)
        for child in node.children:
            visit(child)

    visit(tree.root)
    return result


def test_connect_flow_saves_key_queries_models_and_switches_provider(
    tmp_path: Path, monkeypatch
) -> None:
    import litcode_agent.tui as tui_module

    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.setattr(
        tui_module,
        "fetch_model_list",
        lambda api_key, base_url: ("server-a", "server-b"),
    )

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            app.action_connect()
            await pilot.pause()
            assert app.screen.__class__.__name__ == "ProviderPicker"
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "SecretPrompt"
            await pilot.pause()
            secret = app.screen.query_one("#secret-input", Input)
            secret.value = "sk-connect-key"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause()
                if app.screen.__class__.__name__ == "ModelPicker":
                    break
            assert app.screen.__class__.__name__ == "ModelPicker"
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert app.model.model == "server-a"
            assert app.sessions.active.model.model == "server-a"
            assert app.sessions.active.agent.model_name == "server-a"

            auth = tmp_path / "user-home" / ".local" / "share" / "litcode" / "auth.json"
            store = json.loads(auth.read_text(encoding="utf-8"))
            assert store["credentials"]["DEEPSEEK_API_KEY"] == {
                "type": "api",
                "key": "sk-connect-key",
            }
            assert store["lastClient"]["model"] == "server-a"
            assert store["lastClient"]["baseURL"] == "https://api.deepseek.com"

            session_id = app.session.session_id
            assert app.store.session_info(session_id).model == "server-a"

    asyncio.run(exercise())


def test_connect_custom_endpoint_flow(tmp_path: Path, monkeypatch) -> None:
    import litcode_agent.tui as tui_module

    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.setattr(
        tui_module,
        "fetch_model_list",
        lambda api_key, base_url: (),
    )

    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider_selected("custom")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "CustomEndpointPrompt"
            base = app.screen.query_one("#custom-base", Input)
            base.value = "http://127.0.0.1:8000/v1"
            env = app.screen.query_one("#custom-env", Input)
            env.value = "LITCODE_CUSTOM_API_KEY"
            key = app.screen.query_one("#custom-key", Input)
            key.value = "placeholder"
            app.screen.query_one("#custom-key", Input).focus()
            await pilot.pause()
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause()
                if app.screen.__class__.__name__ == "ModelIDPrompt":
                    break
            assert app.screen.__class__.__name__ == "ModelIDPrompt"
            model_id = app.screen.query_one("#model-id-input", Input)
            model_id.value = "custom-llm"
            await pilot.press("enter")
            await pilot.pause()
            assert app.model.model == "custom-llm"
            assert str(app.model.client.base_url).rstrip("/") == "http://127.0.0.1:8000/v1"
            auth = tmp_path / "user-home" / ".local" / "share" / "litcode" / "auth.json"
            store = json.loads(auth.read_text(encoding="utf-8"))
            assert store["credentials"]["LITCODE_CUSTOM_API_KEY"]["key"] == "placeholder"
            assert store["lastClient"]["baseURL"] == "http://127.0.0.1:8000/v1"

    asyncio.run(exercise())


def test_connect_custom_endpoint_validates_bad_url(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = LitCodeTUI(settings(tmp_path), FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            app._provider_selected("custom")
            await pilot.pause()
            base = app.screen.query_one("#custom-base", Input)
            base.value = "not-a-url"
            key = app.screen.query_one("#custom-key", Input)
            key.value = "x"
            app.screen.query_one("#custom-key", Input).focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "CustomEndpointPrompt"
            error = app.screen.query_one("#custom-error", Static)
            assert "http" in str(error.render())

    asyncio.run(exercise())


def test_unconfigured_tui_guides_connect_and_blocks_messages(tmp_path: Path) -> None:
    from litcode_agent.config import Settings as RealSettings

    unconfigured = RealSettings.load_tui(
        tmp_path, {"HOME": str(tmp_path / "fresh-home")}
    )

    async def exercise() -> None:
        app = LitCodeTUI(unconfigured, FakeModel())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            assert any(
                "connect" in str(widget.render())
                for widget in app.query(".notice").results(Static)
            )
            prompt = app.query_one(PromptArea)
            prompt.text = "hello"
            prompt.action_submit()
            await pilot.pause()
            assert app.store.list_sessions(tmp_path) == ()
            assert any(
                "connect" in str(widget.render())
                for widget in app.query(".notice").results(Static)
            )

    asyncio.run(exercise())
