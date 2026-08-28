"""Textual 全屏交互界面。"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

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
from litcode_agent.hooks import HookRunner
from litcode_agent.model import ModelError, OpenAIChatModel, ToolCall
from litcode_agent.references import (
    ReferenceBundle,
    ReferenceError,
    build_reference_bundle,
    list_workspace_files,
)
from litcode_agent.tools import build_default_registry
from litcode_agent.tools.workspace import Workspace

COMMANDS = (
    ("/help", "显示命令帮助"),
    ("/model", "查询并选择模型"),
    ("/clear", "清空当前对话上下文"),
    ("/exit", "退出 LitCode"),
)


@dataclass(frozen=True, slots=True)
class CompletionContext:
    kind: str
    query: str
    row: int
    start_column: int
    end_column: int


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
        self.workspace = Workspace(settings.workspace)
        self.file_paths: tuple[str, ...] = ()
        self.completion_context: CompletionContext | None = None
        self.completion_values: list[str] = []
        self.streaming_markdown: Markdown | None = None
        self.streaming_buffer = ""
        self.rendered_output: str | None = None
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
            yield OptionList(id="completion", compact=True, markup=False)
            yield Label(
                "输入任务，使用 / 命令或 @ 引用文件 · Ctrl+Enter 发送",
                id="prompt-label",
            )
            yield PromptArea(id="prompt", language=None)
        yield Footer()

    def on_mount(self) -> None:
        self.ui_thread_id = threading.get_ident()
        self._update_status("就绪")
        self._append_notice(
            "会话已启动。输入 /help 查看命令，"
            "工具调用会显示在时间线中。"
        )
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
        self.hide_completions()
        try:
            bundle = build_reference_bundle(
                value,
                self.workspace,
                max_file_chars=self.settings.max_reference_file_chars,
                max_total_chars=self.settings.max_reference_chars,
            )
        except ReferenceError as error:
            self._append_notice(str(error), error=True)
            return
        self._append_user_bundle(bundle)
        self._set_busy(True, "正在启动…")
        self.cancel_requested.clear()
        self.run_worker(
            lambda: self._run_turn(bundle.model_text),
            name="agent-turn",
            group="agent",
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
            self._start_streaming_message()
            self._update_status(f"第 {event.iteration} 轮 · 请求模型")
            return
        if event.kind == "model_delta":
            self._append_stream_delta(event.content or "")
            return
        if event.kind == "model_end":
            self._end_streaming_message(event.content, event.has_tool_calls)
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
            self._append_notice(
                "未知命令；输入 /help 查看帮助。", error=True
            )

    def action_cancel_or_quit(self) -> None:
        if self.busy:
            self.cancel_requested.set()
            self._update_status("正在停止；等待当前阻塞调用返回…")
            return
        self.exit()

    def action_clear_session(self) -> None:
        if self.busy:
            self._append_notice(
                "请先停止或等待当前任务结束。", error=True
            )
            return
        self.session.close("user_clear", "")
        self.session = self.agent.start_session()
        timeline = self.query_one("#timeline", VerticalScroll)
        timeline.remove_children()
        self.tool_bodies.clear()
        self.streaming_markdown = None
        self.rendered_output = None
        self._append_notice("对话上下文已清空。")

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
            self.model.select_model(selected)
            self._append_notice(
                f"已切换到模型 {selected}；对话上下文保持不变。"
            )
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

    def _append_user_bundle(self, bundle: ReferenceBundle) -> None:
        content = bundle.display_text
        if bundle.references:
            paths = "、".join(reference.path for reference in bundle.references)
            content = f"{content}\n\n引用文件：{paths}"
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
        body.update(
            Text(
                f"参数\n{_pretty_json(tool_call.arguments)}"
                f"\n\n结果 · {label}\n{content}"
            )
        )

    def _finish_turn(self, output: str, succeeded: bool) -> None:
        if output != self.rendered_output:
            self._append_assistant(output)
        self.rendered_output = None
        if not succeeded:
            self._append_notice("本轮未正常完成。", error=True)
        self._set_busy(False, "就绪")

    def _finish_with_error(self, message: str) -> None:
        if self.streaming_markdown is not None:
            partial = self.streaming_buffer
            self.streaming_markdown.update(
                f"{partial}\n\n_流式响应中断。_"
                if partial
                else "_模型请求失败。_"
            )
            self.streaming_markdown = None
            self.streaming_buffer = ""
        self._append_notice(message, error=True)
        self._set_busy(False, "错误")

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        prompt = self.query_one(PromptArea)
        prompt.disabled = busy
        self._update_status(status)
        if not busy:
            prompt.focus()
            self.refresh_completions()

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

    def _start_streaming_message(self) -> None:
        self.streaming_buffer = ""
        self.rendered_output = None
        self.streaming_markdown = Markdown(
            "_正在等待模型响应…_", classes="message-assistant"
        )
        self._mount_timeline(self.streaming_markdown)

    def _append_stream_delta(self, content: str) -> None:
        if self.streaming_markdown is None:
            self._start_streaming_message()
        self.streaming_buffer += content
        assert self.streaming_markdown is not None
        self.streaming_markdown.update(f"{self.streaming_buffer} ▍")
        self._update_status("正在接收模型输出…")

    def _end_streaming_message(
        self, content: str | None, has_tool_calls: bool
    ) -> None:
        widget = self.streaming_markdown
        self.streaming_markdown = None
        self.streaming_buffer = ""
        if widget is None:
            return
        if content:
            widget.update(content)
            self.rendered_output = content
            return
        if has_tool_calls:
            widget.remove()
        else:
            widget.update("_模型没有返回文本。_")

    def _build_file_index(self) -> None:
        paths = list_workspace_files(self.workspace)
        self.call_from_thread(self._file_index_ready, paths)

    def _file_index_ready(self, paths: tuple[str, ...]) -> None:
        self.file_paths = paths
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
            candidates = [name for name, description in COMMANDS]
            descriptions = dict(COMMANDS)
            matches = _fuzzy_matches(context.query, candidates, 8)
            labels = [f"{name:<10} {descriptions[name]}" for name in matches]
        else:
            matches = _fuzzy_matches(context.query, self.file_paths, 30)
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
        replacement = f"{value} " if context.kind == "command" else f"@{{{value}}}"
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


def _pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw


def _model_label(model: str, current: str) -> str:
    return f"{model}  （当前）" if model == current else model
