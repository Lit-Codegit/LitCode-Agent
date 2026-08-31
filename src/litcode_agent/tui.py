"""Textual 全屏交互界面。"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.fuzzy import Matcher
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
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from litcode_agent.agent import Agent, AgentEvent, AgentSession
from litcode_agent.config import Settings
from litcode_agent.model import ModelError, OpenAIChatModel, ToolCall
from litcode_agent.pane_layout import PaneBranch, PaneLayout, PaneLeaf, PaneNode
from litcode_agent.references import (
    ReferenceBundle,
    ReferenceError,
    build_reference_bundle,
    list_reference_entries,
)
from litcode_agent.prompt import PromptBuilder
from litcode_agent.session_store import Checkpoint, SessionStore
from litcode_agent.session_workspace import PaneSession, SessionWorkspace
from litcode_agent.skills import SkillCatalog
from litcode_agent.tool_display import tool_result_summary, tool_title
from litcode_agent.tools import build_default_registry
from litcode_agent.tools.workspace import Workspace

@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    handler: str
    aliases: tuple[str, ...] = ()


COMMANDS = (
    CommandSpec("/help", "显示命令帮助", "help"),
    CommandSpec("/model", "查询并选择模型", "model", ("/models",)),
    CommandSpec("/clear", "新建空白会话", "clear"),
    CommandSpec("/sessions", "选择并恢复会话", "sessions", ("/resume",)),
    CommandSpec("/compact", "压缩当前上下文", "compact"),
    CommandSpec("/rewind", "回到历史检查点", "rewind"),
    CommandSpec("/redo", "撤销最近一次 rewind", "redo"),
    CommandSpec("/fork", "从检查点创建分支", "fork"),
    CommandSpec("/split", "按方向创建会话 pane", "split"),
    CommandSpec("/focus", "按方向切换 pane", "focus"),
    CommandSpec("/close-pane", "关闭当前 pane", "close_pane"),
    CommandSpec("/inbox", "查看当前会话收件箱", "inbox"),
    CommandSpec("/exit", "退出 LitCode", "exit", ("/quit",)),
)


@dataclass(frozen=True, slots=True)
class CompletionContext:
    kind: str
    query: str
    row: int
    start_column: int
    end_column: int


@dataclass(slots=True)
class PaneRuntime:
    pane_id: str
    agent: Agent
    model: OpenAIChatModel
    session: AgentSession
    busy: bool = False
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    tool_bodies: dict[str, Static] = field(default_factory=dict)
    tool_cards: dict[str, Collapsible] = field(default_factory=dict)
    streaming_markdown: Markdown | None = None
    streaming_buffer: str = ""
    rendered_output: str | None = None


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

    def on_key(self, event: events.Key) -> None:
        app = self.app
        if not isinstance(app, LitCodeTUI) or not app.completion_visible:
            return
        if event.key in {"up", "down"}:
            app.move_completion(-1 if event.key == "up" else 1)
        elif event.key in {"enter", "tab"}:
            app.accept_completion()
        elif event.key == "escape":
            app.hide_completions()
        else:
            return
        event.prevent_default()
        event.stop()


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

    def __init__(self, command: str, dialog_title: str = "危险命令请求") -> None:
        super().__init__()
        self.command = command
        self.dialog_title = dialog_title

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.dialog_title, classes="dialog-title")
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


class ChoicePicker(ModalScreen[str | None]):
    """Generic keyboard picker used for sessions and checkpoints."""

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, title: str, choices: tuple[tuple[str, str], ...]) -> None:
        super().__init__()
        self.dialog_title = title
        self.choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Label(self.dialog_title, classes="dialog-title")
            yield ListView(
                *(ListItem(Label(label)) for _, label in self.choices),
                initial_index=0,
                id="model-list",
            )
            yield Label("Enter 选择 · Esc 取消", classes="dialog-help")

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.list_view.index is not None:
            self.dismiss(self.choices[event.list_view.index][0])

    def action_cancel(self) -> None:
        self.dismiss(None)


class RewindMode(ModalScreen[str | None]):
    """Choose whether a rewind also restores agent-edited files."""

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label("选择回退范围", classes="dialog-title")
            yield Static("文件回退只覆盖仍与 Agent 写入结果一致的文件。")
            with Horizontal(classes="dialog-buttons"):
                yield Button("仅对话", id="dialogue")
                yield Button("对话和文件", id="files", variant="warning")
                yield Button("取消", id="cancel")

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss(None if event.button.id == "cancel" else event.button.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


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
        Binding("ctrl+w", "pane_leader", "分屏前缀", show=True, priority=True),
        Binding("super+left", "split('left')", "向左分屏", show=False),
        Binding("super+right", "split('right')", "向右分屏", show=False),
        Binding("super+up", "split('up')", "向上分屏", show=False),
        Binding("super+down", "split('down')", "向下分屏", show=False),
        Binding("super+shift+left", "focus_pane('left')", "聚焦左侧", show=False),
        Binding("super+shift+right", "focus_pane('right')", "聚焦右侧", show=False),
        Binding("super+shift+up", "focus_pane('up')", "聚焦上方", show=False),
        Binding("super+shift+down", "focus_pane('down')", "聚焦下方", show=False),
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
    #pane-area, .split-horizontal, .split-vertical {
        height: 1fr;
        width: 1fr;
    }
    .session-pane {
        height: 1fr;
        width: 1fr;
        border: round $secondary;
    }
    .session-pane.pane-active {
        border: round $primary;
    }
    .pane-header {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    .pane-active > .pane-header {
        color: $primary;
        text-style: bold;
    }
    .pane-timeline {
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
    Collapsible.tool-succeeded {
        border-left: thick $success;
    }
    Collapsible.tool-failed {
        border-left: thick $error;
    }
    #composer {
        height: 7;
        padding: 0 1 1 1;
        background: $panel;
        border-top: solid $primary;
    }
    #composer.completing {
        height: 16;
    }
    #completion {
        display: none;
        height: 9;
        margin-bottom: 1;
        border: round $accent;
        background: $panel;
    }
    #completion.visible {
        display: block;
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
    ModelPicker, ConfirmCommand, ChoicePicker, RewindMode {
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
        self.ui_thread_id: int | None = None
        self.shutting_down = False
        self.pending_confirmations: set[threading.Event] = set()
        self.workspace = Workspace(settings.workspace)
        self.skills = SkillCatalog.discover(settings.workspace)
        assert settings.session_database is not None
        self.store = SessionStore(settings.session_database)
        self.file_paths: tuple[str, ...] = ()
        self.directory_paths: tuple[str, ...] = ()
        self.completion_context: CompletionContext | None = None
        self.completion_values: list[str] = []
        self.registry = build_default_registry(
            settings,
            self.confirm_command,
            self.skills,
            self.store,
            self.confirm_session_message,
        )
        self.system_prompt = PromptBuilder(
            settings.workspace,
            settings.max_iterations,
            self.skills.metadata(),
        ).build()
        self.sessions = SessionWorkspace(
            settings,
            model,
            self.registry,
            self.system_prompt,
            self.store,
            self.receive_agent_event,
        )
        first = self.sessions.active
        self.agent = first.agent
        self.panes = {first.pane_id: self._pane_runtime(first)}
        self._pane_leader = False

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        yield Vertical(self._pane_widget(self._active_runtime()), id="pane-area")
        with Vertical(id="composer"):
            yield OptionList(id="completion", compact=True, markup=False)
            yield Label(
                "输入任务，使用 / 命令、@ 文件或 # 会话 · Ctrl+Enter 发送",
                id="prompt-label",
            )
            yield PromptArea(id="prompt", language=None)
        yield Footer()

    def _active_runtime(self) -> PaneRuntime:
        return self.panes[self.active_pane_id]

    @property
    def active_pane_id(self) -> str:
        return self.sessions.active_pane_id

    @property
    def pane_layout(self) -> PaneLayout:
        return self.sessions.layout

    @staticmethod
    def _pane_runtime(pane: PaneSession) -> PaneRuntime:
        return PaneRuntime(pane.pane_id, pane.agent, pane.model, pane.session)

    def _sync_runtime(self, pane: PaneSession) -> PaneRuntime:
        runtime = self.panes[pane.pane_id]
        runtime.agent = pane.agent
        runtime.model = pane.model
        runtime.session = pane.session
        self.agent = pane.agent
        self.model = pane.model
        return runtime

    @property
    def session(self) -> AgentSession:
        return self._active_runtime().session

    @property
    def busy(self) -> bool:
        return self._active_runtime().busy

    def _pane_widget(self, runtime: PaneRuntime) -> Vertical:
        info = self.store.session_info(runtime.session.session_id)
        unread = len(self.store.inbox(runtime.session.session_id))
        unread_label = f" · 未读 {unread}" if unread else ""
        timeline_id = (
            "timeline"
            if runtime.pane_id == "pane-1"
            else f"timeline-{runtime.pane_id}"
        )
        classes = (
            "session-pane pane-active"
            if runtime.pane_id == self.active_pane_id
            else "session-pane"
        )
        return Vertical(
            Static(
                Text(f"{info.alias} · {info.title} · {info.model}{unread_label}"),
                classes="pane-header",
            ),
            VerticalScroll(id=timeline_id, classes="pane-timeline"),
            id=f"view-{runtime.pane_id}",
            classes=classes,
        )

    def _layout_widget(self, node: PaneNode):
        if isinstance(node, PaneLeaf):
            return self._pane_widget(self.panes[node.pane_id])
        children = (self._layout_widget(node.first), self._layout_widget(node.second))
        if node.axis == "horizontal":
            return Horizontal(*children, classes="split-horizontal")
        return Vertical(*children, classes="split-vertical")

    def _rebuild_panes(self) -> None:
        area = self.query_one("#pane-area", Vertical)
        area.remove_children()
        area.mount(self._layout_widget(self.pane_layout.root))
        self.call_after_refresh(self._render_all_pane_histories)

    def _render_all_pane_histories(self) -> None:
        for runtime in self.panes.values():
            self._render_runtime_history(runtime)

    def action_pane_leader(self) -> None:
        self._pane_leader = True
        self._update_status("分屏：按方向键创建 pane")

    def on_key(self, event: events.Key) -> None:
        if not self._pane_leader:
            return
        self._pane_leader = False
        if event.key in {"left", "right", "up", "down"}:
            self.action_split(event.key)
            event.prevent_default()
            event.stop()
            return
        self._update_status("已取消分屏前缀")

    def on_mount(self) -> None:
        self.ui_thread_id = threading.get_ident()
        self._update_status("就绪")
        self._append_notice(
            "会话已启动。输入 /help 查看命令，"
            "工具调用会显示在时间线中。"
        )
        for issue in self.skills.issues:
            self._append_notice(f"Skill 加载失败：{issue}", error=True)
        self.query_one(PromptArea).focus()
        self.run_worker(
            self._build_file_index,
            name="file-index",
            group="file-index",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def on_unmount(self) -> None:
        self.shutting_down = True
        for runtime in self.panes.values():
            runtime.cancel_requested.set()
        for confirmation in self.pending_confirmations:
            confirmation.set()
        self.sessions.close_all()
        self.store.close()

    @on(PromptArea.Submitted)
    def submit_prompt(self, event: PromptArea.Submitted) -> None:
        value = event.value
        if value.startswith("/"):
            self._handle_command(value)
            return
        if self.busy:
            self._append_notice("当前任务仍在运行。", error=True)
            return
        self.hide_completions()
        try:
            bundle = build_reference_bundle(
                value,
                self.workspace,
                max_file_chars=self.settings.max_reference_file_chars,
                max_total_chars=self.settings.max_reference_chars,
                read_roots=self.settings.read_roots,
                session_store=self.store,
                max_session_chars=self.settings.max_session_reference_chars,
            )
        except ReferenceError as error:
            self._append_notice(str(error), error=True)
            return
        for reference in bundle.session_references:
            self.store.record_session_reference(
                self.session.session_id,
                reference.alias,
                reference.updated_at,
                reference.content,
            )
        self._append_user_bundle(bundle)
        runtime = self._active_runtime()
        self._set_pane_busy(runtime, True, "正在启动…")
        runtime.cancel_requested.clear()
        self.run_worker(
            lambda: self._run_turn(bundle.model_text, runtime.pane_id),
            name=f"agent-turn-{runtime.pane_id}",
            group=f"agent-{runtime.pane_id}",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    @on(TextArea.Changed, "#prompt")
    @on(TextArea.SelectionChanged, "#prompt")
    def prompt_updated(self) -> None:
        if not self.busy:
            self.refresh_completions()

    @on(OptionList.OptionSelected, "#completion")
    def completion_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self._insert_completion(int(event.option.id))

    def _run_turn(self, value: str, pane_id: str) -> None:
        runtime = self.panes[pane_id]
        try:
            result = runtime.session.ask(value, runtime.cancel_requested.is_set)
        except ModelError as error:
            self.call_from_thread(self._finish_with_error, str(error), pane_id)
            return
        except Exception as error:  # keep an unexpected worker error visible
            self.call_from_thread(
                self._finish_with_error,
                f"未预期错误：{type(error).__name__}: {error}",
                pane_id,
            )
            return
        self.call_from_thread(
            self._finish_turn, result.output, result.succeeded, pane_id
        )

    def receive_agent_event(self, event: AgentEvent) -> None:
        if self.shutting_down:
            return
        runtime = next(
            (
                item
                for item in self.panes.values()
                if item.session.session_id == event.session_id
            ),
            None,
        )
        if runtime is None:
            return
        if threading.get_ident() == self.ui_thread_id:
            self._render_agent_event(event, runtime)
        else:
            self.call_from_thread(self._render_agent_event, event, runtime)

    def _render_agent_event(self, event: AgentEvent, runtime: PaneRuntime) -> None:
        if event.kind == "model_start":
            self._start_streaming_message(runtime)
            self._update_pane_status(runtime, f"第 {event.iteration} 轮 · 请求模型")
            return
        if event.kind == "model_delta":
            self._append_stream_delta(event.content or "", runtime)
            return
        if event.kind == "model_end":
            self._end_streaming_message(event.content, event.has_tool_calls, runtime)
            return
        if event.kind == "hook_result":
            assert event.hook_execution is not None
            execution = event.hook_execution
            status = "完成" if execution.return_code == 0 else "失败"
            self._append_notice(f"hook {execution.event} · {status}", runtime=runtime)
            return
        assert event.tool_call is not None
        if event.kind == "tool_start":
            self._append_tool(event.tool_call, runtime)
            self._update_pane_status(runtime, f"正在调用工具 · {event.tool_call.name}")
            return
        self._finish_tool(
            event.tool_call,
            event.content or "（无输出）",
            event.is_error,
            runtime,
        )

    def _handle_command(self, command: str) -> None:
        name, _, arguments = command.strip().partition(" ")
        spec = next(
            (item for item in COMMANDS if name == item.name or name in item.aliases),
            None,
        )
        if spec is None:
            self._append_notice(
                "未知命令；输入 /help 查看帮助。", error=True
            )
            return
        handlers = {
            "exit": lambda: self.action_cancel_or_quit(),
            "help": self._show_help,
            "model": self.action_choose_model,
            "clear": self.action_clear_session,
            "sessions": self.action_choose_session,
            "compact": lambda: self.action_compact(arguments),
            "rewind": self.action_rewind,
            "redo": self.action_redo,
            "fork": self.action_fork,
            "split": lambda: self._command_direction(arguments, self.action_split),
            "focus": lambda: self._command_direction(arguments, self.action_focus_pane),
            "close_pane": self.action_close_pane,
            "inbox": self.action_inbox,
        }
        handlers[spec.handler]()

    def _command_direction(self, arguments: str, action) -> None:
        direction = arguments.strip().lower()
        if direction not in {"left", "right", "up", "down"}:
            self._append_notice(
                "方向必须是 left、right、up 或 down。", error=True
            )
            return
        action(direction)

    def _show_help(self) -> None:
        self._append_notice(
            " · ".join(f"{item.name} {item.description}" for item in COMMANDS)
            + " · Ctrl+Enter 发送"
        )

    def action_cancel_or_quit(self) -> None:
        if self.busy:
            self._active_runtime().cancel_requested.set()
            self._update_status("正在停止；等待当前阻塞调用返回…")
            return
        self.exit()

    def action_split(self, direction: str) -> None:
        if direction not in {"left", "right", "up", "down"}:
            self._append_notice(f"未知分屏方向：{direction}", error=True)
            return
        if len(self.panes) >= 4:
            self._append_notice("第一版最多同时打开 4 个 pane。", error=True)
            return
        if any(runtime.busy for runtime in self.panes.values()):
            self._append_notice(
                "等待所有 pane 当前任务结束后再改变布局。", error=True
            )
            return
        pane = self.sessions.split(direction)
        runtime = self._pane_runtime(pane)
        self.panes[pane.pane_id] = runtime
        self.agent = runtime.agent
        self.model = runtime.model
        self._rebuild_panes()
        self._set_pane_busy(runtime, False, "新 pane")

    def action_focus_pane(self, direction: str) -> None:
        if direction not in {"left", "right", "up", "down"}:
            self._append_notice(f"未知焦点方向：{direction}", error=True)
            return
        previous = self.active_pane_id
        pane = self.sessions.focus(direction)
        if pane is None:
            self._append_notice(f"{direction} 方向没有 pane。")
            return
        target = pane.pane_id
        try:
            self.query_one(f"#view-{previous}").remove_class("pane-active")
            self.query_one(f"#view-{target}").add_class("pane-active")
        except Exception:
            pass
        runtime = self._active_runtime()
        self.agent = runtime.agent
        self.model = runtime.model
        prompt = self.query_one(PromptArea)
        prompt.disabled = runtime.busy
        if not runtime.busy:
            prompt.focus()
        self._update_status("运行中" if runtime.busy else "就绪")

    def action_close_pane(self) -> None:
        if len(self.panes) == 1:
            self._append_notice("不能关闭最后一个 pane。", error=True)
            return
        runtime = self._active_runtime()
        if runtime.busy:
            self._append_notice("请先停止当前 pane 的任务。", error=True)
            return
        removed_id, pane = self.sessions.close_active_pane()
        del self.panes[removed_id]
        self._sync_runtime(pane)
        self._rebuild_panes()
        self._update_status("就绪")

    def action_inbox(self) -> None:
        messages = self.sessions.consume_inbox()
        if not messages:
            self._append_notice("当前会话没有未读消息。")
            return
        for message in messages:
            source = self.store.session_info(message.source_session_id)
            self._append_notice(
                f"来自 {source.alias} · {source.title}：\n{message.content}"
            )
        self._update_pane_header(self._active_runtime(), "就绪")

    def action_clear_session(self) -> None:
        if self.busy:
            self._append_notice(
                "请先停止或等待当前任务结束。", error=True
            )
            return
        self.sessions.clear_active()
        self._sync_runtime(self.sessions.active)
        runtime = self._active_runtime()
        timeline = self._timeline(self._active_runtime())
        timeline.remove_children()
        runtime.tool_bodies.clear()
        runtime.tool_cards.clear()
        runtime.streaming_markdown = None
        runtime.rendered_output = None
        self._append_notice("对话上下文已清空。")

    def action_choose_session(self) -> None:
        if self.busy:
            self._append_notice("当前任务结束后才能切换会话。", error=True)
            return
        sessions = self.store.list_sessions(self.settings.workspace)
        choices = tuple(
            (item.id, f"{item.alias} · {item.title} · {item.model}")
            for item in sessions
        )
        if not choices:
            self._append_notice("没有可恢复的会话。")
            return
        self.push_screen(ChoicePicker("恢复会话", choices), self._session_selected)

    def _session_selected(self, identifier: str | None) -> None:
        if identifier is None or identifier == self.session.session_id:
            return
        if self.busy:
            self._append_notice("当前任务结束后才能切换会话。", error=True)
            return
        self._sync_runtime(self.sessions.switch_active(identifier))
        self._render_session_history("已恢复会话")

    def action_compact(self, instructions: str = "") -> None:
        if self.busy:
            self._append_notice("当前任务结束后才能压缩。", error=True)
            return
        self._set_busy(True, "正在压缩上下文…")
        pane_id = self.active_pane_id
        self.run_worker(
            lambda: self._compact_worker(instructions, pane_id),
            name=f"compact-{pane_id}",
            group=f"agent-{pane_id}",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _compact_worker(self, instructions: str, pane_id: str) -> None:
        runtime = self.panes[pane_id]
        try:
            summary = runtime.session.compact(instructions)
        except (ModelError, ValueError) as error:
            self.call_from_thread(self._finish_with_error, str(error), pane_id)
            return
        self.call_from_thread(self._compact_finished, summary, pane_id)

    def _compact_finished(self, summary: str, pane_id: str) -> None:
        runtime = self.panes[pane_id]
        self._append_notice(
            f"上下文已压缩：\n{summary[:1200]}", runtime=runtime
        )
        self._set_pane_busy(runtime, False, "就绪")

    def action_rewind(self) -> None:
        self._choose_checkpoint("选择 rewind 检查点", self._rewind_checkpoint)

    def _rewind_checkpoint(self, identifier: str | None) -> None:
        checkpoint = self._checkpoint(identifier)
        if checkpoint is not None:
            self._pending_checkpoint = checkpoint
            self.push_screen(RewindMode(), self._rewind_mode_selected)

    def _rewind_mode_selected(self, mode: str | None) -> None:
        checkpoint = getattr(self, "_pending_checkpoint", None)
        if checkpoint is None or mode is None:
            return
        try:
            count = self.session.rewind(checkpoint, restore_files=mode == "files")
        except RuntimeError as error:
            self._append_notice(str(error), error=True)
            return
        self._render_session_history(
            f"已回到检查点：{checkpoint.label}；恢复文件 {count} 个"
        )

    def action_redo(self) -> None:
        try:
            count = self.session.redo()
        except RuntimeError as error:
            self._append_notice(str(error), error=True)
            return
        self._render_session_history(f"已撤销 rewind；恢复文件 {count} 个")

    def action_fork(self) -> None:
        self._choose_checkpoint("选择 fork 检查点", self._fork_checkpoint)

    def _fork_checkpoint(self, identifier: str | None) -> None:
        checkpoint = self._checkpoint(identifier)
        if checkpoint is None:
            return
        self.sessions.fork_active(checkpoint)
        self._sync_runtime(self.sessions.active)
        self._render_session_history(f"已从检查点创建分支：{checkpoint.label}")

    def _choose_checkpoint(self, title: str, callback) -> None:
        checkpoints = self.session.checkpoints()
        choices = tuple(
            (item.id, f"{item.label} · {item.id[:8]}") for item in checkpoints
        )
        if not choices:
            self._append_notice("当前会话还没有检查点。")
            return
        self.push_screen(ChoicePicker(title, choices), callback)

    def _checkpoint(self, identifier: str | None) -> Checkpoint | None:
        if identifier is None:
            return None
        return next(
            (item for item in self.session.checkpoints() if item.id == identifier),
            None,
        )

    def _render_session_history(self, notice: str) -> None:
        runtime = self._active_runtime()
        self._render_runtime_history(runtime)
        self._append_notice(notice, runtime=runtime)

    def _render_runtime_history(self, runtime: PaneRuntime) -> None:
        timeline = self._timeline(runtime)
        timeline.remove_children()
        runtime.tool_bodies.clear()
        runtime.tool_cards.clear()
        runtime.streaming_markdown = None
        runtime.streaming_buffer = ""
        runtime.rendered_output = None
        for message in runtime.session.messages[1:]:
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            if message.get("role") == "user":
                self._mount_timeline(
                    Static(Text(_display_user_content(content)), classes="message-user"),
                    runtime,
                )
            elif message.get("role") == "assistant":
                self._append_assistant(content, runtime)

    def action_choose_model(self) -> None:
        if self.busy:
            self._append_notice(
                "当前任务结束后才能切换模型。", error=True
            )
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
            self.sessions.select_model(selected)
            self._append_notice(
                f"已切换到模型 {selected}；对话上下文保持不变。"
            )
        self._update_status("就绪")
        self.query_one(PromptArea).focus()

    def confirm_command(self, command: str) -> bool:
        return self._confirm_action(command, "危险命令请求")

    def confirm_session_message(self, description: str) -> bool:
        return self._confirm_action(description, "跨会话消息确认")

    def _confirm_action(self, content: str, title: str) -> bool:
        finished = threading.Event()
        decision = {"allowed": False}
        self.pending_confirmations.add(finished)

        def resolved(allowed: bool | None) -> None:
            decision["allowed"] = bool(allowed)
            finished.set()

        self.call_from_thread(
            self.push_screen,
            ConfirmCommand(content, title),
            resolved,
        )
        finished.wait()
        self.pending_confirmations.discard(finished)
        return decision["allowed"]

    def _append_user_bundle(self, bundle: ReferenceBundle) -> None:
        content = bundle.display_text
        if bundle.references:
            paths = "、".join(reference.path for reference in bundle.references)
            content = f"{content}\n\n引用文件：{paths}"
        if bundle.session_references:
            aliases = "、".join(
                reference.alias for reference in bundle.session_references
            )
            content = f"{content}\n\n引用会话：{aliases}"
        self._mount_timeline(
            Static(Text(content), classes="message-user"), self._active_runtime()
        )

    def _append_assistant(
        self, content: str, runtime: PaneRuntime | None = None
    ) -> None:
        self._mount_timeline(
            Markdown(content, classes="message-assistant"), runtime
        )

    def _append_notice(
        self,
        content: str,
        *,
        error: bool = False,
        runtime: PaneRuntime | None = None,
    ) -> None:
        classes = "notice notice-error" if error else "notice"
        self._mount_timeline(Static(Text(content), classes=classes), runtime)

    def _append_tool(self, tool_call: ToolCall, runtime: PaneRuntime) -> None:
        body = Static(Text("运行中…"))
        card = Collapsible(
            body,
            title=tool_title(tool_call, "●"),
            collapsed=True,
        )
        runtime.tool_bodies[tool_call.id] = body
        runtime.tool_cards[tool_call.id] = card
        self._mount_timeline(card, runtime)

    def _finish_tool(
        self,
        tool_call: ToolCall,
        content: str,
        is_error: bool,
        runtime: PaneRuntime,
    ) -> None:
        body = runtime.tool_bodies.get(tool_call.id)
        card = runtime.tool_cards.get(tool_call.id)
        if body is None or card is None:
            self._append_notice(content, error=is_error, runtime=runtime)
            return
        card.title = tool_title(tool_call, "✗" if is_error else "✓")
        card.add_class("tool-failed" if is_error else "tool-succeeded")
        body.update(Text(tool_result_summary(content, is_error)))
        for item in self.panes.values():
            self._update_pane_header(item, "运行中" if item.busy else "就绪")

    def _finish_turn(self, output: str, succeeded: bool, pane_id: str) -> None:
        runtime = self.panes[pane_id]
        if output != runtime.rendered_output:
            self._append_assistant(output, runtime)
        runtime.rendered_output = None
        if not succeeded:
            self._append_notice("本轮未正常完成。", error=True, runtime=runtime)
        self._set_pane_busy(runtime, False, "就绪")

    def _finish_with_error(self, message: str, pane_id: str | None = None) -> None:
        runtime = self.panes[pane_id] if pane_id is not None else self._active_runtime()
        if runtime.streaming_markdown is not None:
            partial = runtime.streaming_buffer
            runtime.streaming_markdown.update(
                f"{partial}\n\n_流式响应中断。_"
                if partial
                else "_模型请求失败。_"
            )
            runtime.streaming_markdown = None
            runtime.streaming_buffer = ""
        self._append_notice(message, error=True, runtime=runtime)
        self._set_pane_busy(runtime, False, "错误")

    def _set_busy(self, busy: bool, status: str) -> None:
        self._set_pane_busy(self._active_runtime(), busy, status)

    def _set_pane_busy(
        self, runtime: PaneRuntime, busy: bool, status: str
    ) -> None:
        runtime.busy = busy
        self._update_pane_header(runtime, status)
        if runtime.pane_id != self.active_pane_id:
            return
        prompt = self.query_one(PromptArea)
        prompt.disabled = busy
        self._update_status(status)
        if not busy:
            prompt.focus()
            self.refresh_completions()

    def _update_pane_status(self, runtime: PaneRuntime, status: str) -> None:
        self._update_pane_header(runtime, status)
        if runtime.pane_id == self.active_pane_id:
            self._update_status(status)

    def _update_pane_header(self, runtime: PaneRuntime, status: str) -> None:
        try:
            header = self.query_one(f"#view-{runtime.pane_id} .pane-header", Static)
        except Exception:
            return
        info = self.store.session_info(runtime.session.session_id)
        header.update(Text(f"{info.alias} · {info.title} · {status}"))

    def _update_status(self, activity: str) -> None:
        info = self.store.session_info(self.session.session_id)
        self.query_one("#status", Static).update(
            Text(
                f"{self.settings.workspace}  ·  "
                f"{info.alias}  ·  {self.settings.model_profile} / "
                f"{self.model.model}  ·  {activity}"
            )
        )

    def _timeline(self, runtime: PaneRuntime) -> VerticalScroll:
        timeline_id = (
            "timeline"
            if runtime.pane_id == "pane-1"
            else f"timeline-{runtime.pane_id}"
        )
        return self.query_one(f"#{timeline_id}", VerticalScroll)

    def _mount_timeline(
        self,
        widget: Static | Markdown | Collapsible,
        runtime: PaneRuntime | None = None,
    ) -> None:
        runtime = runtime or self._active_runtime()
        timeline = self._timeline(runtime)
        timeline.mount(widget)
        timeline.scroll_end(animate=False)

    def _start_streaming_message(self, runtime: PaneRuntime) -> None:
        runtime.streaming_buffer = ""
        runtime.rendered_output = None
        runtime.streaming_markdown = Markdown(
            "_正在等待模型响应…_", classes="message-assistant"
        )
        self._mount_timeline(runtime.streaming_markdown, runtime)

    def _append_stream_delta(self, content: str, runtime: PaneRuntime) -> None:
        if runtime.streaming_markdown is None:
            self._start_streaming_message(runtime)
        runtime.streaming_buffer += content
        assert runtime.streaming_markdown is not None
        runtime.streaming_markdown.update(f"{runtime.streaming_buffer} ▍")
        self._update_pane_status(runtime, "正在接收模型输出…")

    def _end_streaming_message(
        self,
        content: str | None,
        has_tool_calls: bool,
        runtime: PaneRuntime,
    ) -> None:
        widget = runtime.streaming_markdown
        runtime.streaming_markdown = None
        runtime.streaming_buffer = ""
        if widget is None:
            return
        if content:
            widget.update(content)
            runtime.rendered_output = content
            return
        if has_tool_calls:
            widget.remove()
        else:
            widget.update("_模型没有返回文本。_")

    def _build_file_index(self) -> None:
        entries = list_reference_entries(self.workspace, self.settings.read_roots)
        self.call_from_thread(
            self._file_index_ready,
            entries.files,
            entries.directories,
        )

    def _file_index_ready(
        self,
        files: tuple[str, ...],
        directories: tuple[str, ...],
    ) -> None:
        self.file_paths = files
        self.directory_paths = directories
        self.refresh_completions()

    @property
    def completion_visible(self) -> bool:
        return self.query_one("#completion", OptionList).has_class("visible")

    def refresh_completions(self) -> None:
        prompt = self.query_one(PromptArea)
        context = _completion_context(prompt)
        if context is None:
            self.hide_completions()
            return
        if context.kind == "command":
            candidates = [item.name for item in COMMANDS]
            descriptions = {item.name: item.description for item in COMMANDS}
            matches = _fuzzy_matches(context.query, candidates, 8)
            labels = [f"{name:<10} {descriptions[name]}" for name in matches]
        elif context.kind == "session":
            sessions = self.store.list_sessions(self.settings.workspace)
            candidates = [item.alias for item in sessions]
            descriptions = {
                item.alias: f"{item.title} · {item.model}" for item in sessions
            }
            matches = _fuzzy_matches(context.query, candidates, 12)
            labels = [f"{alias}  {descriptions[alias]}" for alias in matches]
        else:
            paths = [*self.directory_paths, *self.file_paths]
            matches = _fuzzy_matches(context.query, paths, 30)
            if context.query.endswith("/"):
                matches = [path for path in matches if path != context.query]
            labels = matches
        if not matches:
            self.hide_completions()
            return
        self.completion_context = context
        self.completion_values = matches
        options = [
            Option(Text(label), id=str(index))
            for index, label in enumerate(labels)
        ]
        popup = self.query_one("#completion", OptionList)
        popup.clear_options().add_options(options)
        popup.highlighted = 0
        popup.add_class("visible")
        self.query_one("#composer").add_class("completing")

    def hide_completions(self) -> None:
        self.completion_context = None
        self.completion_values.clear()
        self.query_one("#completion", OptionList).remove_class("visible")
        self.query_one("#composer").remove_class("completing")

    def move_completion(self, offset: int) -> None:
        popup = self.query_one("#completion", OptionList)
        if not self.completion_values:
            return
        current = popup.highlighted or 0
        popup.highlighted = (current + offset) % len(self.completion_values)

    def accept_completion(self) -> None:
        popup = self.query_one("#completion", OptionList)
        if popup.highlighted is not None:
            self._insert_completion(popup.highlighted)

    def _insert_completion(self, index: int) -> None:
        context = self.completion_context
        if context is None or not 0 <= index < len(self.completion_values):
            return
        value = self.completion_values[index]
        is_directory = context.kind == "file" and value.endswith("/")
        if context.kind == "command":
            replacement = f"{value} "
        elif context.kind == "session":
            replacement = f"#{{{value}}}"
        elif is_directory:
            replacement = f"@{{{value}"
        else:
            replacement = f"@{{{value}}}"
        prompt = self.query_one(PromptArea)
        end = (context.row, context.end_column)
        prompt.replace(
            replacement,
            (context.row, context.start_column),
            end,
        )
        prompt.move_cursor(
            (context.row, context.start_column + len(replacement))
        )
        self.hide_completions()
        if is_directory:
            self.refresh_completions()


def _completion_context(prompt: PromptArea) -> CompletionContext | None:
    row, column = prompt.selection.end
    lines = prompt.text.split("\n")
    if row >= len(lines):
        return None
    prefix = lines[row][:column]
    if row == 0:
        command = re.fullmatch(r"(\s*)/([^\s]*)", prefix)
        if command:
            return CompletionContext(
                "command",
                f"/{command.group(2)}",
                row,
                len(command.group(1)),
                column,
            )
    reference = re.search(r"(?:^|\s)(@\{?([^}\s]*))$", prefix)
    if reference:
        return CompletionContext(
            "file",
            reference.group(2),
            row,
            reference.start(1),
            column,
        )
    session_reference = re.search(r"(?:^|\s)(#\{?([^}\s]*))$", prefix)
    if session_reference:
        return CompletionContext(
            "session",
            session_reference.group(2),
            row,
            session_reference.start(1),
            column,
        )
    return None


def _fuzzy_matches(
    query: str,
    candidates: tuple[str, ...] | list[str],
    limit: int,
) -> list[str]:
    if not query:
        return list(candidates[:limit])
    matcher = Matcher(query)
    scored = [
        (matcher.match(candidate), candidate)
        for candidate in candidates
    ]
    return [
        candidate
        for score, candidate in sorted(scored, key=lambda item: (-item[0], item[1]))
        if score > 0
    ][:limit]


def run_tui(settings: Settings, model: OpenAIChatModel) -> int:
    LitCodeTUI(settings, model).run()
    return 0


def _model_label(model: str, current: str) -> str:
    return f"{model}  （当前）" if model == current else model


def _display_user_content(content: str) -> str:
    marker = "\n\n以下是用户明确引用的本地文件快照。"
    return content.partition(marker)[0]
