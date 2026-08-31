from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from textual.widgets import Collapsible, Markdown, Static

from litcode_agent.agent import AgentEvent
from litcode_agent.config import Settings
from litcode_agent.model import AssistantTurn, ModelDelta, ToolCall
from litcode_agent.tui import (
    COMMANDS,
    ConfirmCommand,
    LitCodeTUI,
    ModelPicker,
    PromptArea,
    run_tui,
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
            header = str(app.query_one(".pane-header", Static).render())
            assert "新会话" in header
            assert "model-a" in header
            assert app.store.session_info(app.session.session_id).alias not in header
            assert len(app.query("#status")) == 0
            assert app.query_one(PromptArea).has_focus
            assert any(
                "会话已启动" in str(widget.render())
                for widget in app.query(".notice").results(Static)
            )

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
            for _ in range(30):
                await pilot.pause(0.02)
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
            for _ in range(30):
                await pilot.pause(0.02)
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


def test_run_tui_leaves_mouse_to_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    options: dict[str, object] = {}

    def fake_run(app: LitCodeTUI, **kwargs: object) -> None:
        options.update(kwargs)
        app.store.close()

    monkeypatch.setattr(LitCodeTUI, "run", fake_run)

    assert run_tui(settings(tmp_path), FakeModel()) == 0  # type: ignore[arg-type]
    assert options == {"mouse": False}


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
            await pilot.press("enter")
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
            await pilot.press("enter")
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

            await pilot.press("enter")
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
            for _ in range(30):
                await pilot.pause(0.02)
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
            for _ in range(30):
                await pilot.pause(0.02)
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

            app._handle_command("/focus left")
            prompt = app.query_one(PromptArea)
            prompt.text = "慢任务"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause(0.02)
                if slow_started.is_set():
                    break

            assert app.panes["pane-1"].busy
            app._handle_command("/focus right")
            assert not prompt.disabled
            prompt.text = "快任务"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause(0.02)
                if not app.panes["pane-2"].busy:
                    break

            assert app.panes["pane-1"].streaming_buffer == "慢"
            fast_timeline = app.query_one("#timeline-pane-2")
            assert len(list(fast_timeline.query(Markdown))) == 1
            assert app.panes["pane-2"].session.messages[-1]["content"] == "快完成"

            release_slow.set()
            for _ in range(30):
                await pilot.pause(0.02)
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
            for _ in range(30):
                await pilot.pause(0.02)
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
            for _ in range(30):
                await pilot.pause(0.02)
                if (
                    first_delta.is_set()
                    and app._active_runtime().streaming_buffer == "流"
                ):
                    break

            assert app.busy
            assert app._active_runtime().streaming_buffer == "流"
            assert app._active_runtime().streaming_markdown is not None

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
