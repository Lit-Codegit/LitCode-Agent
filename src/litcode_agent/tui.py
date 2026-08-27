"""Textual 全屏交互界面。"""

from __future__ import annotations

import json
import threading

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
    TextArea,
)

from litcode_agent.agent import Agent, AgentEvent, AgentSession
from litcode_agent.config import Settings
from litcode_agent.hooks import HookRunner
from litcode_agent.model import ModelError, OpenAIChatModel, ToolCall
from litcode_agent.tools import build_default_registry


class PromptArea(TextArea):
    """以 Ctrl+Enter 提交的固定多行输入框。"""

    BINDINGS = [Binding("ctrl+enter", "submit", "发送", show=True)]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit(self) -> None:
        value = self.text.strip()
        if value:
            self.text = ""
            self.post_message(self.Submitted(value))


class ModelPicker(ModalScreen[str | None]):
    """从服务端模型列表中选择一项。"""

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, models: tuple[str, ...], current: str) -> None:
        super().__init__()
        self.models = models
        self.current = current

    def compose(self) -> ComposeResult:
        initial = self.models.index(self.current) if self.current in self.models else 0
        items = [
            ListItem(Label(Text(_model_label(model, self.current))))
            for model in self.models
        ]
        with Vertical(id="model-dialog"):
            yield Label("选择模型", classes="dialog-title")
            yield ListView(*items, initial_index=initial, id="model-list")
            yield Label("Enter 选择 · Esc 取消", classes="dialog-help")

    @on(ListView.Selected)
    def select_model(self, event: ListView.Selected) -> None:
        if event.list_view.index is not None:
            self.dismiss(self.models[event.list_view.index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmCommand(ModalScreen[bool]):
    """危险命令确认窗口。"""

    BINDINGS = [Binding("escape", "deny", "拒绝")]

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label("危险命令请求", classes="dialog-title")
            yield Static(Text(self.command), id="confirm-command")
            with Horizontal(classes="dialog-buttons"):
                yield Button("拒绝", id="deny", variant="default")
                yield Button("允许", id="allow", variant="error")

    @on(Button.Pressed, "#allow")
    def allow(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def deny(self) -> None:
        self.dismiss(False)

    def action_deny(self) -> None:
        self.dismiss(False)


class LitCodeTUI(App[None]):
    """LitCode Agent 的常驻全屏界面。"""

    TITLE = "LitCode Agent"
    BINDINGS = [
        Binding(
            "ctrl+c",
            "cancel_or_quit",
            "停止 / 退出",
            show=True,
            priority=True,
        ),
        Binding("ctrl+l", "clear_session", "清空", show=True),
        Binding("f2", "choose_model", "模型", show=True),
    ]
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    #status {
        height: 3;
        padding: 0 1;
        content-align: left middle;
        background: $panel;
        border-bottom: solid $primary;
    }
    #timeline {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    .message-user, .message-assistant, .notice, Collapsible {
        margin: 0 0 1 0;
        padding: 1 2;
    }
    .message-user {
        background: $primary 18%;
        border-left: thick $primary;
    }
    .message-assistant {
        background: $success 10%;
        border-left: thick $success;
    }
    .notice {
        color: $text-muted;
        padding: 0 1;
    }
    .notice-error {
        color: $error;
    }
    Collapsible {
        background: $boost;
        border-left: thick $accent;
    }
    #composer {
        height: 7;
        padding: 0 1 1 1;
        background: $panel;
        border-top: solid $primary;
    }
    #prompt-label {
        height: 1;
        color: $text-muted;
    }
    #prompt {
        height: 5;
        border: tall $primary;
    }
    Footer {
        height: 1;
    }
    ModelPicker, ConfirmCommand {
        align: center middle;
        background: $background 70%;
    }
    #model-dialog, #confirm-dialog {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $panel;
        border: round $primary;
    }
    #model-list {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }
    .dialog-title {
        text-style: bold;
        color: $primary;
    }
    .dialog-help {
        color: $text-muted;
    }
    #confirm-command {
        margin: 1 0;
        padding: 1;
        background: $boost;
        max-height: 12;
        overflow-y: auto;
    }
    .dialog-buttons {
        height: 3;
        align-horizontal: right;
    }
    .dialog-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, settings: Settings, model: OpenAIChatModel) -> None:
        super().__init__()
        self.settings = settings
        self.model = model
        self.cancel_requested = threading.Event()
        self.ui_thread_id: int | None = None
        self.shutting_down = False
        self.busy = False
        self.pending_confirmations: set[threading.Event] = set()
        self.tool_bodies: dict[str, Static] = {}
        self.agent = Agent(
            model,
            build_default_registry(settings, self.confirm_command),
            settings.max_iterations,
            self.receive_agent_event,
            HookRunner(settings.workspace, settings.hooks),
        )
        self.session: AgentSession = self.agent.start_session()

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        yield VerticalScroll(id="timeline")
        with Vertical(id="composer"):
            yield Label(
                "输入任务或 /help · Ctrl+Enter 发送 · Enter 换行",
                id="prompt-label",
            )
            yield PromptArea(id="prompt", language=None)
        yield Footer()

    def on_mount(self) -> None:
        self.ui_thread_id = threading.get_ident()
        self._update_status("就绪")
        self._append_notice(
            "会话已启动。输入 /help 查看命令，工具调用会显示在时间线中。"
        )
        self.query_one(PromptArea).focus()

    def on_unmount(self) -> None:
        self.shutting_down = True
        self.cancel_requested.set()
        for confirmation in self.pending_confirmations:
            confirmation.set()
        self.session.close("user_exit", "")

    @on(PromptArea.Submitted)
    def submit_prompt(self, event: PromptArea.Submitted) -> None:
        value = event.value
        if value.startswith("/"):
            self._handle_command(value)
            return
        if self.busy:
            self._append_notice("当前任务仍在运行。", error=True)
            return
        self._append_user(value)
        self._set_busy(True, "正在启动…")
        self.cancel_requested.clear()
        self.run_worker(
            lambda: self._run_turn(value),
            name="agent-turn",
            group="agent",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _run_turn(self, value: str) -> None:
        try:
            result = self.session.ask(value, self.cancel_requested.is_set)
        except ModelError as error:
            self.call_from_thread(self._finish_with_error, str(error))
            return
        except Exception as error:  # keep an unexpected worker error visible
            self.call_from_thread(
                self._finish_with_error,
                f"未预期错误：{type(error).__name__}: {error}",
            )
            return
        self.call_from_thread(self._finish_turn, result.output, result.succeeded)

    def receive_agent_event(self, event: AgentEvent) -> None:
        if self.shutting_down:
            return
        if threading.get_ident() == self.ui_thread_id:
            self._render_agent_event(event)
        else:
            self.call_from_thread(self._render_agent_event, event)

    def _render_agent_event(self, event: AgentEvent) -> None:
        if event.kind == "model_start":
            self._update_status(f"第 {event.iteration} 轮 · 请求模型")
            return
        if event.kind == "hook_result":
            assert event.hook_execution is not None
            execution = event.hook_execution
            status = "完成" if execution.return_code == 0 else "失败"
            self._append_notice(f"hook {execution.event} · {status}")
            return
        assert event.tool_call is not None
        if event.kind == "tool_start":
            self._append_tool(event.tool_call)
            self._update_status(f"正在调用工具 · {event.tool_call.name}")
            return
        self._finish_tool(
            event.tool_call,
            event.content or "（无输出）",
            event.is_error,
        )

    def _handle_command(self, command: str) -> None:
        if command in {"/exit", "/quit"}:
            self.action_cancel_or_quit()
        elif command == "/help":
            self._append_notice(
                "/model 选择模型 · /clear 清空上下文 · /exit 退出 · "
                "Ctrl+Enter 发送"
            )
        elif command in {"/model", "/models"}:
            self.action_choose_model()
        elif command == "/clear":
            self.action_clear_session()
        else:
            self._append_notice("未知命令；输入 /help 查看帮助。", error=True)

    def action_cancel_or_quit(self) -> None:
        if self.busy:
            self.cancel_requested.set()
            self._update_status("正在停止；等待当前阻塞调用返回…")
            return
        self.exit()

    def action_clear_session(self) -> None:
        if self.busy:
            self._append_notice("请先停止或等待当前任务结束。", error=True)
            return
        self.session.close("user_clear", "")
        self.session = self.agent.start_session()
        timeline = self.query_one("#timeline", VerticalScroll)
        timeline.remove_children()
        self.tool_bodies.clear()
        self._append_notice("对话上下文已清空。")

    def action_choose_model(self) -> None:
        if self.busy:
            self._append_notice("当前任务结束后才能切换模型。", error=True)
            return
        self._set_busy(True, "正在查询模型…")
        self.run_worker(
            self._query_models,
            name="model-query",
            group="model-query",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _query_models(self) -> None:
        try:
            models = self.model.list_models()
        except ModelError as error:
            self.call_from_thread(self._finish_with_error, str(error))
            return
        self.call_from_thread(self._open_model_picker, models)

    def _open_model_picker(self, models: tuple[str, ...]) -> None:
        self._set_busy(False, "就绪")
        if not models:
            self._append_notice("API 没有返回可选模型。", error=True)
            return
        self.push_screen(ModelPicker(models, self.model.model), self._model_selected)

    def _model_selected(self, selected: str | None) -> None:
        if selected and selected != self.model.model:
            self.model.select_model(selected)
            self._append_notice(f"已切换到模型 {selected}；对话上下文保持不变。")
        self._update_status("就绪")
        self.query_one(PromptArea).focus()

    def confirm_command(self, command: str) -> bool:
        finished = threading.Event()
        decision = {"allowed": False}
        self.pending_confirmations.add(finished)

        def resolved(allowed: bool | None) -> None:
            decision["allowed"] = bool(allowed)
            finished.set()

        self.call_from_thread(
            self.push_screen,
            ConfirmCommand(command),
            resolved,
        )
        finished.wait()
        self.pending_confirmations.discard(finished)
        return decision["allowed"]

    def _append_user(self, content: str) -> None:
        self._mount_timeline(Static(Text(content), classes="message-user"))

    def _append_assistant(self, content: str) -> None:
        self._mount_timeline(Markdown(content, classes="message-assistant"))

    def _append_notice(self, content: str, *, error: bool = False) -> None:
        classes = "notice notice-error" if error else "notice"
        self._mount_timeline(Static(Text(content), classes=classes))

    def _append_tool(self, tool_call: ToolCall) -> None:
        body = Static(Text(_pretty_json(tool_call.arguments)))
        self.tool_bodies[tool_call.id] = body
        self._mount_timeline(
            Collapsible(
                body,
                title=f"工具 · {tool_call.name}",
                collapsed=False,
            )
        )

    def _finish_tool(
        self, tool_call: ToolCall, content: str, is_error: bool
    ) -> None:
        body = self.tool_bodies.get(tool_call.id)
        if body is None:
            self._append_notice(content, error=is_error)
            return
        label = "失败" if is_error else "完成"
        body.update(Text(f"参数\n{_pretty_json(tool_call.arguments)}\n\n结果 · {label}\n{content}"))

    def _finish_turn(self, output: str, succeeded: bool) -> None:
        self._append_assistant(output)
        if not succeeded:
            self._append_notice("本轮未正常完成。", error=True)
        self._set_busy(False, "就绪")

    def _finish_with_error(self, message: str) -> None:
        self._append_notice(message, error=True)
        self._set_busy(False, "错误")

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        prompt = self.query_one(PromptArea)
        prompt.disabled = busy
        self._update_status(status)
        if not busy:
            prompt.focus()

    def _update_status(self, activity: str) -> None:
        self.query_one("#status", Static).update(
            Text(
                f"{self.settings.workspace}  ·  "
                f"{self.settings.model_profile} / {self.model.model}  ·  {activity}"
            )
        )

    def _mount_timeline(self, widget: Static | Markdown | Collapsible) -> None:
        timeline = self.query_one("#timeline", VerticalScroll)
        timeline.mount(widget)
        timeline.scroll_end(animate=False)


def run_tui(settings: Settings, model: OpenAIChatModel) -> int:
    LitCodeTUI(settings, model).run()
    return 0


def _pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw


def _model_label(model: str, current: str) -> str:
    return f"{model}  （当前）" if model == current else model
