"""Textual 全屏交互界面。"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

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
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    OptionList,
    Static,
    TextArea,
    Tree,
)
from textual.widgets.option_list import Option

from litcode_agent.agent import Agent, AgentEvent, AgentSession
from litcode_agent.config import Settings
from litcode_agent.model import (
    Message as ModelMessage,
    ModelError,
    OpenAIChatModel,
    ToolCall,
)
from litcode_agent.pane_layout import PaneBranch, PaneLayout, PaneLeaf, PaneNode
from litcode_agent.references import (
    ReferenceBundle,
    ReferenceError,
    build_reference_bundle,
    list_reference_entries,
    SESSION_REFERENCE_PATTERN,
)
from litcode_agent.prompt import PromptBuilder
from litcode_agent.session_store import Checkpoint, SessionInfo, SessionStore, SessionTurn
from litcode_agent.session_runtime import SessionRuntime
from litcode_agent.session_workspace import PaneSession, SessionWorkspace
from litcode_agent.skills import SkillCatalog
from litcode_agent.tool_display import (
    change_result_summary,
    tool_result_summary,
    tool_title,
)
from litcode_agent.tools import build_default_registry
from litcode_agent.tools.base import FileChange, ToolError
from litcode_agent.tools.question import MAX_OPTIONS, QuestionSpec
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
    CommandSpec("/new", "新建会话并切换当前 pane", "new"),
    CommandSpec("/clear", "新建空白会话", "clear"),
    CommandSpec(
        "/history",
        "浏览会话树并挂载到当前 pane（原会话转入后台）",
        "history",
        ("/sessions", "/tree", "/resume"),
    ),
    CommandSpec("/compact", "压缩当前上下文", "compact"),
    CommandSpec("/rewind", "回到历史检查点", "rewind"),
    CommandSpec("/redo", "撤销最近一次 rewind", "redo"),
    CommandSpec("/fork", "从检查点创建分支", "fork"),
    CommandSpec("/split", "按方向创建 pane，可选挂载会话", "split"),
    CommandSpec("/focus", "按方向切换 pane", "focus"),
    CommandSpec("/close-pane", "关闭当前 pane", "close_pane"),
    CommandSpec("/nohup", "卸载当前会话并让它在后台继续", "nohup"),
    CommandSpec("/subagent", "创建一个子会话并按需后台运行", "subagent"),
    CommandSpec("/inbox", "查看当前会话收件箱", "inbox"),
    CommandSpec("/queue", "查看或管理当前会话队列", "queue"),
    CommandSpec("/exit", "退出 LitCode", "exit", ("/quit",)),
)


LIST_CARD_LINE_THRESHOLD = 12
PICKER_LABEL_CELLS = 60
PICKER_GUIDE_DEPTH = 3


@dataclass(frozen=True, slots=True)
class CompletionContext:
    kind: str
    query: str
    row: int
    start_column: int
    end_column: int


@dataclass(slots=True)
class _PendingQuestion:
    """One ask_user request waiting on user input from a worker thread."""

    session_id: str
    questions: list[QuestionSpec]
    answers: list[list[str]] | None = None
    finished: threading.Event = field(default_factory=threading.Event)


@dataclass(slots=True)
class PaneRuntime:
    pane_id: str
    pane_slot: int
    agent: Agent
    model: OpenAIChatModel
    session: AgentSession | None
    busy: bool = False
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    tool_bodies: dict[str, PagedOutput] = field(default_factory=dict)
    tool_cards: dict[str, Collapsible] = field(default_factory=dict)
    streaming_markdown: Markdown | None = None
    streaming_buffer: str = ""
    rendered_output: str | None = None
    prompt_history: list[str] = field(default_factory=list)
    prompt_history_index: int | None = None
    prompt_draft: str = ""

    @property
    def empty(self) -> bool:
        return self.session is None


class PromptArea(TextArea):
    """Enter 发送、组合键换行，并支持 shell 风格历史。"""

    BINDINGS = [
        Binding("enter", "submit", "发送", show=True),
        Binding(
            "shift+enter,ctrl+enter",
            "newline",
            "换行",
            show=False,
        ),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit(self) -> None:
        value = self.text.strip()
        if value:
            app = self.app
            if isinstance(app, LitCodeTUI):
                app.record_prompt(value)
            self.text = ""
            self.post_message(self.Submitted(value))

    def action_newline(self) -> None:
        start, end = self.selection
        result = self.replace(
            "\n", start, end, maintain_selection_offset=False
        )
        self.move_cursor(result.end_location)


    def on_key(self, event: events.Key) -> None:
        app = self.app
        if not isinstance(app, LitCodeTUI):
            return
        if app.pane_leader_active and app.consume_pane_leader_key(event.key):
            event.prevent_default()
            event.stop()
            return
        if app.completion_visible:
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
            return
        if event.key == "enter":
            self.action_submit()
            event.prevent_default()
            event.stop()
            return
        if event.key == "up" and self.cursor_at_first_line:
            if app.navigate_prompt_history(-1):
                event.prevent_default()
                event.stop()
            return
        if event.key == "down" and self.cursor_at_last_line:
            if app.navigate_prompt_history(1):
                event.prevent_default()
                event.stop()
            return
        if event.key not in {"up", "down"}:
            app.reset_prompt_history_navigation()


class PagedOutput(Static):
    """有界内翻页正文：只显示一页，用 j/k 或上下键翻页。"""

    PAGE_LINE_LIMIT = 18
    BINDINGS = [
        Binding("j,down,page_down", "next_page", "下一页", show=False),
        Binding("k,up,page_up", "prev_page", "上一页", show=False),
    ]

    can_focus = True

    def __init__(self, content: str = "") -> None:
        super().__init__(classes="paged-output")
        self._source_lines: list[str] = []
        self._page_index = 0
        self.set_content(content)

    def set_content(self, content: str) -> None:
        self._source_lines = content.splitlines() or ["（无输出）"]
        self._page_index = 0
        self.update(self._page_text())

    @property
    def page_count(self) -> int:
        return max(1, -(-len(self._source_lines) // self.PAGE_LINE_LIMIT))

    def _page_text(self) -> Text:
        start = self._page_index * self.PAGE_LINE_LIMIT
        lines = self._source_lines[start : start + self.PAGE_LINE_LIMIT]
        content = Text()
        for index, line in enumerate(lines):
            if index:
                content.append("\n")
            content.append(line, style=_diff_style(line))
        if self.page_count > 1:
            content.append(
                f"\n第 {self._page_index + 1}/{self.page_count} 页 · "
                "j/k 翻页",
                style="dim",
            )
        return content

    def action_next_page(self) -> None:
        if self.page_count > 1 and self._page_index < self.page_count - 1:
            self._page_index += 1
            self.update(self._page_text())

    def action_prev_page(self) -> None:
        if self._page_index > 0:
            self._page_index -= 1
            self.update(self._page_text())

    def on_click(self, event: events.Click) -> None:
        self.focus()
        event.stop()

    def on_focus(self) -> None:
        self.add_class("paged-focused")

    def on_blur(self) -> None:
        self.remove_class("paged-focused")


class PaneDivider(Static):
    """A one-cell divider whose drag is translated into a layout ratio delta."""

    def __init__(self, target_pane_id: str, axis: str) -> None:
        super().__init__(classes="pane-divider")
        self.target_pane_id = target_pane_id
        self.axis = axis
        self._dragging = False
        self._last_coordinate = 0

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._dragging = True
        self._last_coordinate = (
            event.screen_x if self.axis == "horizontal" else event.screen_y
        )
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._dragging or not isinstance(self.app, LitCodeTUI):
            return
        coordinate = (
            event.screen_x if self.axis == "horizontal" else event.screen_y
        )
        delta_pixels = coordinate - self._last_coordinate
        if delta_pixels:
            parent_size = self.parent.size
            denominator = (
                parent_size.width if self.axis == "horizontal" else parent_size.height
            )
            if denominator > 0:
                self.app.resize_pane(
                    self.target_pane_id,
                    self.axis,
                    delta_pixels / denominator,
                )
            self._last_coordinate = coordinate
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
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


class HistoryPicker(ModalScreen[str | None]):
    """树状历史会话选择器：筛选 + Enter 挂载，原会话转为后台运行。

    接口形状借鉴 OpenCode 的会话选择弹窗（搜索框 + 有界列表 + Enter 提交），
    但保留 LitCode 的会话树层级：子会话缩进展示，候选数有界并提示总数。
    """

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(
        self,
        rows: tuple[tuple[int, SessionInfo], ...],
        *,
        mounted: dict[str, int],
        running: set[str],
        terminal_id: str,
        query: str = "",
        current_session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.mounted = mounted
        self.running = running
        self.terminal_id = terminal_id
        self.filter_text = query
        self.current_session_id = current_session_id
        self._pending_cursor = None

    def compose(self) -> ComposeResult:
        session_tree = Tree("会话", id="history-tree")
        session_tree.show_root = False
        session_tree.guide_depth = 3
        with Vertical(id="history-dialog"):
            yield Label("选择历史会话", classes="dialog-title")
            yield session_tree
            yield Input(
                value=self.filter_text,
                placeholder="筛选：编号 / 标题 / 模型",
                id="history-filter",
            )
            yield Label("", classes="dialog-help", id="history-help")

    def on_mount(self) -> None:
        self._refresh_tree()

    @on(Tree.NodeSelected, "#history-tree")
    def tree_selected(self, event: Tree.NodeSelected) -> None:
        identifier = event.node.data if isinstance(event.node.data, str) else None
        self.dismiss(identifier or self._first_visible_data())

    @on(Input.Changed, "#history-filter")
    def filter_changed(self, event: Input.Changed) -> None:
        self.filter_text = event.value
        self._refresh_tree()

    @on(Input.Submitted, "#history-filter")
    def filter_submitted(self, event: Input.Submitted) -> None:
        tree = self.query_one("#history-tree", Tree)
        identifier = tree.cursor_node.data if tree.cursor_node is not None else None
        self.dismiss(
            identifier if isinstance(identifier, str) else self._first_visible_data()
        )

    def _first_visible_data(self) -> str | None:
        tree = self.query_one("#history-tree", Tree)

        def first(node) -> str | None:
            if isinstance(node.data, str):
                return node.data
            for child in node.children:
                found = first(child)
                if found is not None:
                    return found
            return None

        return first(tree.root)

    def _refresh_tree(self) -> None:
        tree = self.query_one("#history-tree", Tree)
        tree.clear()
        visible = self._filtered_infos()
        children: dict[str | None, list[str]] = {}
        for depth, info in visible:
            children.setdefault(info.parent_id, []).append(info.id)
        infos = {info.id: info for _, info in visible}

        def add_node(parent, info: SessionInfo, depth: int):
            node = parent.add(
                _history_label(
                    info,
                    self.mounted,
                    self.running,
                    self.current_session_id,
                    depth,
                ),
                data=info.id,
            )
            for identifier in children.get(info.id, []):
                add_node(node, infos[identifier], depth + 1)
            return node

        first_node = None
        for identifier in children.get(None, []):
            node = add_node(tree.root, infos[identifier], 0)
            first_node = first_node or node
        self._pending_cursor = first_node
        if first_node is not None:
            self.call_after_refresh(self._apply_cursor)
        self.query_one("#history-help", Label).update(
            f"共 {len(self.rows)} 个会话 · 显示 {len(visible)} 个 · Enter 选择 · Esc 取消"
        )

    def _apply_cursor(self) -> None:
        """把光标落到第一个节点；Tree 的行映射在访问时惰性构建。"""

        tree = self.query_one("#history-tree", Tree)
        if self._pending_cursor is None:
            return
        tree.cursor_line = self._pending_cursor._line
        tree.move_cursor(self._pending_cursor)
        if tree.cursor_line < 0:
            tree.cursor_line = 0

    def _filtered_infos(self) -> list[tuple[int, SessionInfo]]:
        filter_text = self.filter_text.strip()
        if not filter_text:
            return list(self.rows)
        matcher = Matcher(filter_text)
        matched = {
            info.id
            for _, info in self.rows
            if matcher.match(info.alias) > 0
            or matcher.match(info.title) > 0
            or matcher.match(info.model) > 0
        }
        parents = {info.id: info.parent_id for _, info in self.rows}
        keep: set[str] = set()
        for identifier in matched:
            current: str | None = identifier
            while current is not None and current not in keep:
                keep.add(current)
                current = parents.get(current)
        return [
            (depth, info)
            for depth, info in self.rows
            if info.id in keep
        ]

    def action_cancel(self) -> None:
        self.dismiss(None)


class QuestionPrompt(ModalScreen[list[list[str]] | None]):
    """收集 ask_user 工具的问题答案。

    交互形态借鉴 OpenCode 的 question 视图：顶部问题标签页、数字键直选、
    j/k 移动、Enter 确认、自定义回答输入、Esc 拒绝。单个单选题选中即提交；
    多个问题时用 Tab 或 h/l 切换问题（最后一页是确认页）。
    """

    BINDINGS = (
        [Binding("escape", "cancel", "取消"), Binding("enter", "confirm", "确认")]
        + [
            Binding("left,h", "prev_tab", "上一题", show=False),
            Binding("right,l", "next_tab", "下一题", show=False),
            Binding("tab", "next_tab", "下一题", show=False),
            Binding("j,down", "option_down", "下一项", show=False),
            Binding("k,up", "option_up", "上一项", show=False),
        ]
        + [
            Binding(str(number), f"pick_option({number - 1})", "选项", show=False)
            for number in range(1, MAX_OPTIONS + 2)
        ]
    )

    def __init__(self, questions: list[QuestionSpec], title: str) -> None:
        super().__init__()
        self.questions = questions
        self.title = title
        self.answers: list[list[str]] = [[] for _ in questions]
        self.tab: int = 0
        self.custom_active = False

    @property
    def confirm_tab(self) -> int:
        return len(self.questions)

    @property
    def submitting_on_pick(self) -> bool:
        return len(self.questions) == 1 and not self.questions[0].multiple

    def compose(self) -> ComposeResult:
        with Vertical(id="question-dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Static("", id="question-tabs")
            yield Static("", id="question-text")
            yield OptionList(id="question-options", markup=False)
            yield Input(
                placeholder="自定义回答，Enter 提交",
                id="question-custom",
            )
            yield Label(
                "数字选择 · j/k 移动 · Enter 确认 · h/l 或 Tab 切换问题 · Esc 拒绝",
                classes="dialog-help",
            )

    def on_mount(self) -> None:
        self._refresh_tab()

    @on(OptionList.OptionSelected, "#question-options")
    def option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self._pick(int(event.option.id))

    @on(Input.Changed, "#question-custom")
    def custom_changed(self, event: Input.Changed) -> None:
        if event.value:
            self.custom_active = True

    @on(Input.Submitted, "#question-custom")
    def custom_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self.answers[self.tab] = [text]
        self.custom_active = False
        if self.submitting_on_pick:
            self.dismiss(list(self.answers))
            return
        self._advance_tab()

    def _refresh_tab(self) -> None:
        self.query_one("#question-tabs", Static).update(
            Text(self._tab_labels(), style="dim")
        )
        options_list = self.query_one("#question-options", OptionList)
        text_widget = self.query_one("#question-text", Static)
        custom_input = self.query_one("#question-custom", Input)
        if self.tab == self.confirm_tab:
            text_widget.update(Text(self._summary_text(), style="bold"))
            options_list.clear_options()
            options_list.display = False
            custom_input.display = False
            return
        question = self.questions[self.tab]
        text_widget.update(Text(f"{question.header}\n{question.question}"))
        options = list(question.options)
        if question.custom:
            options.append(("✎ 自定义回答", "输入你自己的答案"))
        options_list.display = True
        options_list.clear_options()
        options_list.add_options(
            Option(
                Text(("✓ " if label in self.tab_answers else "") + label),
                id=str(index),
            )
            for index, (label, description) in enumerate(options)
        )
        options_list.highlighted = 0
        if self.custom_active:
            custom_input.display = True
            custom_input.focus()
        else:
            custom_input.display = False
            options_list.focus()

    @property
    def tab_answers(self) -> list[str]:
        return self.answers[self.tab] if self.tab < len(self.answers) else []

    def _tab_labels(self) -> str:
        labels = [
            f"[{index + 1}] {question.header}"
            for index, question in enumerate(self.questions)
        ]
        labels.append("[确认]")
        return "  ".join(
            label
            if index != self.tab
            else f"<{label}>"
            for index, label in enumerate(labels)
        )

    def _summary_text(self) -> str:
        lines = []
        for index, question in enumerate(self.questions):
            selected = self.answers[index]
            answer = ", ".join(selected) if selected else "未回答"
            lines.append(f"{index + 1}. {question.header}：{question.question}\n   → {answer}")
        return "\n".join(lines)

    def _pick(self, index: int) -> None:
        if self.tab == self.confirm_tab:
            return
        question = self.questions[self.tab]
        if index < len(question.options):
            label = question.options[index][0]
            if question.multiple:
                current = list(self.answers[self.tab])
                self.answers[self.tab] = (
                    [item for item in current if item != label]
                    if label in current
                    else current + [label]
                )
                self._refresh_tab()
                return
            self.answers[self.tab] = [label]
            if self.submitting_on_pick:
                self.dismiss(list(self.answers))
                return
            self._advance_tab()
            return
        if question.custom and index == len(question.options):
            self.custom_active = True
            self._refresh_tab()

    def _advance_tab(self) -> None:
        self.tab = min(self.tab + 1, self.confirm_tab)
        self._refresh_tab()

    def action_prev_tab(self) -> None:
        if self.tab > 0:
            self.tab -= 1
            self._refresh_tab()

    def action_next_tab(self) -> None:
        if self.tab < self.confirm_tab:
            self.tab += 1
            self._refresh_tab()

    def action_pick_option(self, number: int) -> None:
        self._pick(number)

    def action_option_down(self) -> None:
        options = self.query_one("#question-options", OptionList)
        if options.option_count:
            options.highlighted = ((options.highlighted or 0) + 1) % options.option_count

    def action_option_up(self) -> None:
        options = self.query_one("#question-options", OptionList)
        if options.option_count:
            options.highlighted = (options.highlighted or 1) - 1

    def action_confirm(self) -> None:
        if self.tab == self.confirm_tab:
            self.dismiss(list(self.answers))

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
        Binding(
            "ctrl+w",
            "pane_leader",
            "后按方向分屏",
            show=True,
            priority=True,
        ),
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
    .pane-divider {
        background: $secondary;
        width: 1;
        height: 1fr;
    }
    .pane-divider:hover {
        background: $accent;
    }
    .split-vertical > .pane-divider {
        width: 1fr;
        height: 1;
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
        background: $primary 10%;
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
    ModelPicker, ConfirmCommand, ChoicePicker, RewindMode, HistoryPicker,
    QuestionPrompt {
        align: center middle;
        background: $background 80%;
    }
    #model-dialog, #confirm-dialog, #history-dialog, #question-dialog {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $panel;
        border: round $primary;
    }
    #history-tree {
        height: auto;
        max-height: 24;
        background: $boost;
        margin: 1 0 0 0;
        border: round $accent;
    }
    #history-filter {
        margin-top: 1;
    }
    #question-tabs {
        height: 1;
        color: $text-muted;
    }
    #question-text {
        margin: 1 0;
        text-style: bold;
    }
    #question-options {
        height: auto;
        max-height: 16;
        margin: 1 0;
    }
    #question-custom {
        height: 1;
        margin: 0 0 1 0;
        padding: 0 2;
        border: none;
        background: $boost;
        width: 100%;
    }
    .paged-output {
        background: $boost;
    }
    .paged-output.paged-focused {
        background: $accent 20%;
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
        self.pending_questions: dict[str, _PendingQuestion] = {}
        self.workspace = Workspace(settings.workspace)
        self.skills = SkillCatalog.discover(settings.workspace)
        assert settings.session_database is not None
        self.store = SessionStore(settings.session_database)
        self.system_prompt = PromptBuilder(
            settings.workspace,
            settings.max_iterations,
            self.skills.metadata(),
        ).build()
        self.runtime = SessionRuntime(
            self.store,
            settings.workspace,
            system_prompt=self.system_prompt,
            event_sink=self._runtime_event,
        )
        self.file_paths: tuple[str, ...] = ()
        self.directory_paths: tuple[str, ...] = ()
        self.completion_context: CompletionContext | None = None
        self.completion_values: list[str] = []
        self._pending_checkpoint: Checkpoint | None = None
        self.registry = build_default_registry(
            settings,
            confirm=self.confirm_command,
            skills=self.skills,
            store=self.store,
            confirm_session_message=self.confirm_session_message,
            confirm_session_read=self.confirm_session_read,
            runtime=self.runtime,
            confirm_session_control=self.confirm_session_control,
            ask_user=self.ask_user,
        )
        self.sessions = SessionWorkspace(
            settings,
            model,
            self.registry,
            self.system_prompt,
            self.store,
            self.receive_agent_event,
            runtime=self.runtime,
        )
        self.runtime.session_factory = self.sessions.session_factory
        first = self.sessions.active
        self.agent = first.agent
        self.panes = {first.pane_id: self._pane_runtime(first)}
        self._pane_layout_generation = 0
        self._pane_leader_until = 0.0
        self._exit_armed_until = 0.0
        self.running_sessions: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Vertical(self._pane_widget(self._active_runtime()), id="pane-area")
        with Vertical(id="composer"):
            yield OptionList(id="completion", compact=True, markup=False)
            yield Label(
                "Enter 发送（运行中自动排队） · Shift+Enter 换行 · ↑↓ 历史 · / @ # 补全",
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
        runtime = PaneRuntime(
            pane.pane_id, pane.pane_slot, pane.agent, pane.model, pane.session
        )
        if pane.session is not None:
            runtime.prompt_history = _message_prompt_history(pane.session.messages)
        return runtime

    def _sync_runtime(self, pane: PaneSession) -> PaneRuntime:
        runtime = self.panes[pane.pane_id]
        session_changed = (
            runtime.session is None
            or pane.session is None
            or runtime.session.session_id != pane.session.session_id
        )
        runtime.agent = pane.agent
        runtime.model = pane.model
        runtime.session = pane.session
        if session_changed and pane.session is not None:
            runtime.prompt_history = _message_prompt_history(pane.session.messages)
            runtime.prompt_history_index = None
            runtime.prompt_draft = ""
        self.agent = pane.agent
        self.model = pane.model
        return runtime

    @property
    def session(self) -> AgentSession:
        """Return the active session, materializing an empty pane on demand.

        UI actions call ``_ensure_active_session`` explicitly.  This property
        keeps the small public test/application surface convenient for callers
        that intentionally ask for the active conversation.
        """

        runtime = self._ensure_active_session()
        assert runtime.session is not None
        return runtime.session

    @property
    def busy(self) -> bool:
        return self._active_runtime().busy

    def _pane_widget(self, runtime: PaneRuntime) -> Vertical:
        if runtime.session is None:
            header_text = f"{runtime.pane_slot}  空窗格 · 输入消息开始"
        else:
            info = self.store.session_info(runtime.session.session_id)
            unread = len(self.store.inbox(runtime.session.session_id))
            unread_label = f" · 未读 {unread}" if unread else ""
            header_text = f"{runtime.pane_slot}  {_session_label(info)}{unread_label}"
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
                Text(header_text),
                classes="pane-header",
            ),
            VerticalScroll(id=timeline_id, classes="pane-timeline"),
            id=f"view-{runtime.pane_id}",
            classes=classes,
        )

    def _layout_widget(self, node: PaneNode):
        if isinstance(node, PaneLeaf):
            return self._pane_widget(self.panes[node.pane_id])
        first = self._layout_widget(node.first)
        second = self._layout_widget(node.second)
        first.styles.width = f"{node.ratio}fr"
        second.styles.width = f"{1.0 - node.ratio}fr"
        divider = PaneDivider(_first_leaf(node.first), node.axis)
        if node.axis == "horizontal":
            return Horizontal(first, divider, second, classes="split-horizontal")
        first.styles.width = "1fr"
        second.styles.width = "1fr"
        first.styles.height = f"{node.ratio}fr"
        second.styles.height = f"{1.0 - node.ratio}fr"
        return Vertical(first, divider, second, classes="split-vertical")

    def resize_pane(self, pane_id: str, axis: str, delta: float) -> None:
        """Apply a bounded divider drag and rebuild only the pane view."""

        if axis not in {"horizontal", "vertical"}:
            raise ValueError(f"unknown pane axis: {axis}")
        self.pane_layout.resize(pane_id, delta)
        self._rebuild_panes()

    def _rebuild_panes(self) -> None:
        self._pane_layout_generation += 1
        generation = self._pane_layout_generation
        area = self.query_one("#pane-area", Vertical)
        area.remove_children()
        area.mount(self._layout_widget(self.pane_layout.root))
        self.call_after_refresh(self._render_all_pane_histories, generation)

    def _render_all_pane_histories(self, generation: int, attempt: int = 0) -> None:
        if generation != self._pane_layout_generation:
            return
        timeline_ids = [
            "timeline" if runtime.pane_slot == 1 else f"timeline-{runtime.pane_id}"
            for runtime in self.panes.values()
        ]
        if any(len(self.query(f"#{timeline_id}")) == 0 for timeline_id in timeline_ids):
            if attempt < 5:
                self.call_after_refresh(
                    self._render_all_pane_histories, generation, attempt + 1
                )
            return
        for runtime in self.panes.values():
            self._render_runtime_history(runtime)

    def action_pane_leader(self) -> None:
        now = time.monotonic()
        if self.pane_leader_active:
            self.action_cycle_pane()
        else:
            self._update_status(
                "方向键分屏 · h/j/k/l 聚焦 · 再按 Ctrl+W 轮换"
            )
        self._pane_leader_until = now + 1.2
        self.set_timer(1.2, self._reset_pane_leader)

    def _reset_pane_leader(self) -> None:
        if time.monotonic() < self._pane_leader_until:
            return
        self._pane_leader_until = 0.0

    def _finish_pane_leader(self) -> None:
        self._pane_leader_until = 0.0

    @property
    def pane_leader_active(self) -> bool:
        return time.monotonic() <= self._pane_leader_until

    def consume_pane_leader_key(self, key: str) -> bool:
        if not self.pane_leader_active:
            return False
        self._finish_pane_leader()
        if key in {"left", "right", "up", "down"}:
            self.action_split(key)
            return True
        focus_directions = {
            "h": "left",
            "j": "down",
            "k": "up",
            "l": "right",
        }
        if key in focus_directions:
            self.action_focus_pane(focus_directions[key])
            return True
        if key == "w":
            self.action_cycle_pane()
            return True
        self._update_status("就绪")
        return True

    def on_key(self, event: events.Key) -> None:
        if not self.pane_leader_active:
            return
        if self.consume_pane_leader_key(event.key):
            event.prevent_default()
            event.stop()

    def on_mount(self) -> None:
        self.ui_thread_id = threading.get_ident()
        self._append_notice(
            "会话已启动。输入 /help 查看命令，"
            "工具调用会显示在时间线中。"
        )
        self._append_notice(
            "分屏：Ctrl+W 后按方向键，或输入 /split right。"
            "多数 macOS 终端不会把 Cmd 组合键传给应用。"
        )
        self._append_notice(
            "切换 pane：Ctrl+W 后按 h/j/k/l 聚焦，按 w 轮换；"
            "也可输入 /focus left。"
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
        for pending in self.pending_questions.values():
            pending.answers = None
            pending.finished.set()
        self.runtime.close()
        self.sessions.close_all()
        self.store.close()

    @on(PromptArea.Submitted)
    def submit_prompt(self, event: PromptArea.Submitted) -> None:
        value = event.value
        if value.startswith("/"):
            self._handle_command(value)
            return
        if (
            self.settings.session_read_policy == "deny"
            and SESSION_REFERENCE_PATTERN.search(value)
        ):
            self._append_notice("当前配置禁止跨会话读取。", error=True)
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
        self._ensure_active_session()
        for reference in bundle.session_references:
            self.store.record_session_reference(
                self.session.session_id,
                reference.alias,
                reference.updated_at,
                reference.content,
            )
        self._append_user_bundle(bundle)
        runtime = self._active_runtime()
        # User messages use the same durable mailbox as Agent messages.  The
        # pane remains a view; a background/child Session can consume its own
        # queue even when no pane is mounted.
        assert runtime.session is not None
        already_running = runtime.busy
        self.runtime.register(runtime.session.session_id, runtime.session)
        if not already_running:
            self._set_pane_busy(runtime, True, "排队中…")
            self.running_sessions.add(runtime.session.session_id)
        try:
            self.runtime.submit(runtime.session.session_id, bundle.model_text)
        except Exception as error:
            self.running_sessions.discard(runtime.session.session_id)
            self._set_pane_busy(runtime, False, "错误")
            self._append_notice(str(error), error=True, runtime=runtime)
            return
        if already_running:
            queued = self.store.queue(runtime.session.session_id)
            position = sum(1 for item in queued if item.status == "queued")
            self._append_notice(
                f"消息已加入队列（第 {position} 条待执行）。", runtime=runtime
            )

    @on(TextArea.Changed, "#prompt")
    @on(TextArea.SelectionChanged, "#prompt")
    def prompt_updated(self) -> None:
        self.refresh_completions()

    @on(OptionList.OptionSelected, "#completion")
    def completion_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self._insert_completion(int(event.option.id))

    def _runtime_for_session(self, session_id: str) -> PaneRuntime | None:
        return next(
            (
                runtime
                for runtime in self.panes.values()
                if runtime.session is not None
                and runtime.session.session_id == session_id
            ),
            None,
        )

    def _ensure_active_session(self) -> PaneRuntime:
        """Materialize a root Session only when an Empty Pane receives input."""

        runtime = self._active_runtime()
        if runtime.session is not None:
            return runtime
        prompt_history = list(runtime.prompt_history)
        self.sessions.materialize(runtime.pane_id)
        self._sync_runtime(self.sessions.active)
        runtime.prompt_history = prompt_history
        self._reset_pane_view()
        self._update_pane_header(runtime, "就绪")
        return runtime

    def receive_agent_event(self, event: AgentEvent) -> None:
        if self.shutting_down:
            return
        runtime = next(
            (
                item
                for item in self.panes.values()
                if item.session is not None
                and item.session.session_id == event.session_id
            ),
            None,
        )
        if runtime is None:
            return
        if threading.get_ident() == self.ui_thread_id:
            self._render_agent_event(event, runtime)
        else:
            self.call_from_thread(self._render_agent_event, event, runtime)

    def _runtime_event(
        self, kind: str, session_id: str, turn: SessionTurn | None
    ) -> None:
        """Project background actor lifecycle into the mounted pane when present."""

        if self.shutting_down:
            return
        if kind in {"turn_finished", "turn_failed"} and turn is not None:
            if threading.get_ident() == self.ui_thread_id:
                self._runtime_turn_finished(turn)
            else:
                self.call_from_thread(self._runtime_turn_finished, turn)
            return
        runtime = self._runtime_for_session(session_id)
        if runtime is None:
            return
        text = {
            "turn_started": "会话轮次已开始",
            "turn_finished": "会话轮次已完成",
            "turn_failed": "会话轮次失败",
            "paused": "会话已暂停",
            "resumed": "会话已恢复",
        }.get(kind, kind)
        if threading.get_ident() == self.ui_thread_id:
            if kind == "turn_started":
                self._runtime_turn_started(runtime)
            else:
                self._append_notice(text, runtime=runtime)
                self._update_pane_header(runtime, "就绪")
            return
        if kind == "turn_started":
            self.call_from_thread(self._runtime_turn_started, runtime)
            return
        self.call_from_thread(self._append_notice, text, runtime=runtime)
        self.call_from_thread(
            self._update_pane_header,
            runtime,
            "运行中" if kind == "turn_started" else "就绪",
        )

    def _runtime_turn_started(self, runtime: PaneRuntime) -> None:
        assert runtime.session is not None
        self.running_sessions.add(runtime.session.session_id)
        self._set_pane_busy(runtime, True, "运行中")

    def _runtime_turn_finished(self, turn: SessionTurn) -> None:
        session_id = turn.session_id
        self.running_sessions.discard(session_id)
        runtime = self._runtime_for_session(session_id)
        if runtime is None:
            return
        if turn.output and turn.output != runtime.rendered_output:
            self._append_assistant(turn.output, runtime)
        runtime.rendered_output = None
        status = turn.status
        if status != "completed":
            if turn.reason == "max_iterations":
                self._append_notice(
                    "已达到迭代上限；以上是模型收尾输出，可继续追问或 /compact。"
                    if turn.output
                    else "已达到迭代上限，模型没有给出收尾输出。",
                    error=True,
                    runtime=runtime,
                )
            else:
                self._append_notice("本轮未正常完成。", error=True, runtime=runtime)
        self._set_pane_busy(runtime, False, "就绪" if status == "completed" else "错误")

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
            event.file_change,
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
            "exit": self.action_exit,
            "help": self._show_help,
            "model": self.action_choose_model,
            "new": self.action_new_session,
            "clear": self.action_clear_session,
            "history": lambda: self.action_history(arguments),
            "compact": lambda: self.action_compact(arguments),
            "rewind": self.action_rewind,
            "redo": self.action_redo,
            "fork": self.action_fork,
            "split": lambda: self._command_split(arguments),
            "focus": lambda: self._command_direction(arguments, self.action_focus_pane),
            "close_pane": self.action_close_pane,
            "nohup": self.action_nohup,
            "subagent": lambda: self.action_subagent(arguments),
            "inbox": self.action_inbox,
            "queue": lambda: self.action_queue(arguments),
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

    def _command_split(self, arguments: str) -> None:
        direction, _, reference = arguments.strip().partition(" ")
        if direction.lower() not in {"left", "right", "up", "down"}:
            self._append_notice(
                "方向必须是 left、right、up 或 down。", error=True
            )
            return
        reference = reference.strip()
        session_id = None
        if reference:
            try:
                session_id = self.store.session_id_for_reference(
                    self.settings.workspace, reference.lstrip("#{").rstrip("}")
                )
            except KeyError:
                self._append_notice(f"找不到会话：{reference}", error=True)
                return
        self.action_split(direction.lower(), session_id=session_id)

    def _show_help(self) -> None:
        self._append_notice(
            " · ".join(f"{item.name} {item.description}" for item in COMMANDS)
            + " · Enter 发送 · Shift+Enter 换行"
        )

    def action_exit(self) -> None:
        """/exit 与 /quit：立即带上运行的 pane 状态退出（无需二次确认）。"""

        for runtime in self.panes.values():
            runtime.cancel_requested.set()
        self.exit()

    def action_cancel_or_quit(self) -> None:
        now = time.monotonic()
        if now <= self._exit_armed_until:
            for runtime in self.panes.values():
                runtime.cancel_requested.set()
            self.exit()
            return
        self._exit_armed_until = now + 1.5
        self.set_timer(1.5, self._reset_exit_arm)
        if self.busy:
            active = self._active_runtime()
            active.cancel_requested.set()
            try:
                self.runtime.cancel_turn(active.session.session_id)
            except (KeyError, RuntimeError):
                pass
            self._update_status("正在停止；再次 Ctrl+C 退出")
            return
        self._update_status("再次 Ctrl+C 退出")

    def _reset_exit_arm(self) -> None:
        if time.monotonic() < self._exit_armed_until:
            return
        self._exit_armed_until = 0.0
        self._update_status("运行中" if self.busy else "就绪")

    def action_split(self, direction: str, session_id: str | None = None) -> None:
        if direction not in {"left", "right", "up", "down"}:
            self._append_notice(f"未知分屏方向：{direction}", error=True)
            return
        if len(self.panes) >= 4:
            self._append_notice("第一版最多同时打开 4 个 pane。", error=True)
            return
        if session_id is not None:
            mounted = self.sessions.pane_for_session(session_id)
            if mounted is not None:
                self._append_notice(
                    f"会话已挂载在 {mounted.pane_slot} 号 pane，已聚焦。"
                )
                self._session_selected(session_id)
                return
        if any(runtime.busy for runtime in self.panes.values()):
            self._append_notice(
                "等待所有 pane 当前任务结束后再改变布局。", error=True
            )
            return
        pane = self.sessions.split(direction, session_id=session_id)
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
        self.query_one(PromptArea).focus()
        self._update_status("运行中" if runtime.busy else "就绪")

    def action_cycle_pane(self) -> None:
        if len(self.panes) == 1:
            self._append_notice("当前只有一个 pane。")
            return
        previous = self.active_pane_id
        pane = self.sessions.focus_next()
        try:
            self.query_one(f"#view-{previous}").remove_class("pane-active")
            self.query_one(f"#view-{pane.pane_id}").add_class("pane-active")
        except Exception:
            pass
        runtime = self._sync_runtime(pane)
        self.query_one(PromptArea).focus()
        self._update_status("运行中" if runtime.busy else "就绪")

    def record_prompt(self, value: str) -> None:
        runtime = self._active_runtime()
        if not runtime.prompt_history or runtime.prompt_history[-1] != value:
            runtime.prompt_history.append(value)
        runtime.prompt_history_index = None
        runtime.prompt_draft = ""

    def navigate_prompt_history(self, direction: int) -> bool:
        runtime = self._active_runtime()
        history = runtime.prompt_history
        if not history or direction not in {-1, 1}:
            return False
        prompt = self.query_one(PromptArea)
        index = runtime.prompt_history_index
        if direction == -1:
            if index is None:
                runtime.prompt_draft = prompt.text
                index = len(history) - 1
            elif index > 0:
                index -= 1
        elif index is None:
            return False
        elif index < len(history) - 1:
            index += 1
        else:
            index = None
        runtime.prompt_history_index = index
        prompt.text = runtime.prompt_draft if index is None else history[index]
        prompt.move_cursor(prompt.document.end)
        return True

    def reset_prompt_history_navigation(self) -> None:
        runtime = self._active_runtime()
        runtime.prompt_history_index = None

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
        lines = []
        for message in messages:
            source = self.store.session_info(message.source_session_id)
            lines.append(f"来自 {source.alias} · {source.title}：\n{message.content}")
        self._list_or_notice("收件箱", "\n\n".join(lines))
        self._update_pane_header(self._active_runtime(), "就绪")

    def action_queue(self, arguments: str = "") -> None:
        """Let the user inspect and mutate queued-but-not-started messages."""

        runtime = self._active_runtime()
        if runtime.session is None:
            self._append_notice("当前 pane 为空，没有消息队列。")
            return
        parts = arguments.split()
        messages = self.store.queue(runtime.session.session_id, include_finished=True)
        if not parts:
            if not messages:
                self._append_notice("当前会话队列为空。")
                return
            lines = [
                f"{message.id[:8]} · {message.status} · {message.content[:120]}"
                for message in messages
            ]
            self._list_or_notice("消息队列", "\n".join(lines), runtime=runtime)
            return
        action = parts[0].lower()
        if action not in {"cancel", "up", "down", "move"}:
            self._append_notice(
                "用法：/queue [cancel|up|down|move] <消息 ID> [before ID]",
                error=True,
            )
            return
        if len(parts) < 2:
            self._append_notice("缺少消息 ID。", error=True)
            return
        message_id = _find_queue_message(messages, parts[1])
        if message_id is None:
            self._append_notice(f"找不到队列消息：{parts[1]}", error=True)
            return
        try:
            if action == "cancel":
                self.store.cancel_queued_message(message_id)
            elif action in {"up", "down"}:
                self.store.move_queued_message(
                    runtime.session.session_id, message_id, -1 if action == "up" else 1
                )
            else:
                if len(parts) < 3:
                    raise ValueError("move 需要 before 消息 ID")
                before = _find_queue_message(messages, parts[2])
                if before is None:
                    raise ValueError(f"找不到 before 消息：{parts[2]}")
                self.store.reorder_queued_message(
                    runtime.session.session_id, message_id, before
                )
        except (KeyError, ValueError) as error:
            self._append_notice(str(error), error=True)
            return
        self._append_notice("队列已更新。", runtime=runtime)

    def action_clear_session(self) -> None:
        if self.busy:
            self._append_notice(
                "请先停止或等待当前任务结束。", error=True
            )
            return
        self.sessions.clear_active()
        self._sync_runtime(self.sessions.active)
        self._reset_pane_view()
        self._append_notice("对话上下文已清空。")

    def action_new_session(self) -> None:
        self.sessions.new_active()
        self._sync_runtime(self.sessions.active)
        self._reset_pane_view()
        self._set_pane_busy(self._active_runtime(), False, "就绪")
        self._append_notice(
            "已创建新会话，当前 pane 已切换；原会话没有结束，仍在后台，可用 /sessions 返回。"
        )

    def action_nohup(self) -> None:
        """Detach the mounted Session while leaving its actor alive."""

        runtime = self._active_runtime()
        if runtime.session is None:
            self._append_notice("当前 pane 已经是空窗格。")
            return
        current = runtime.session.session_id
        if self.busy:
            self._append_notice("当前会话已卸载；正在运行的 turn 会继续。")
        if len(self.panes) > 1:
            removed_id, pane = self.sessions.close_active_pane()
            del self.panes[removed_id]
            self._sync_runtime(pane)
            self._rebuild_panes()
        else:
            self.sessions.detach_active()
            runtime.session = None
            runtime.busy = False
            self._reset_pane_view()
            self._update_pane_header(runtime, "空窗格")
        info = self.store.session_info(current)
        self._append_notice(
            f"会话 {info.alias} 已转入后台；可用 /sessions 重新挂载。"
        )

    def action_subagent(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            self._append_notice("用法：/subagent <目标 prompt>", error=True)
            return
        self._ensure_active_session()
        try:
            info = self.runtime.create_subagent_session(
                self.session.session_id, prompt, start=True
            )
        except Exception as error:
            self._append_notice(str(error), error=True)
            return
        self._append_notice(
            f"已创建子会话 {info.alias}；它会在自己的队列中运行，可从 /sessions 挂载。"
        )

    def action_history(self, arguments: str = "") -> None:
        """Open the interactive session-tree picker.

        允许在任意时刻打开：当前 pane 的任务不会被中断，切换后它转入后台
        继续运行。``arguments`` 作为预填的筛选关键字。
        """

        rows = self.store.session_tree(self.settings.workspace)
        if not rows:
            self._append_notice("当前还没有会话。")
            return
        active = self.sessions.active
        current_id = active.session.session_id if active.session is not None else None
        self.push_screen(
            HistoryPicker(
                rows,
                mounted=self.sessions.mounted_sessions(),
                running=set(self.running_sessions),
                terminal_id=self.sessions.terminal_id,
                query=arguments.strip(),
                current_session_id=current_id,
            ),
            self._session_selected,
        )

    def _reset_pane_view(self) -> None:
        runtime = self._active_runtime()
        timeline = self._timeline(runtime)
        timeline.remove_children()
        runtime.tool_bodies.clear()
        runtime.tool_cards.clear()
        runtime.streaming_markdown = None
        runtime.rendered_output = None

    def _session_selected(self, identifier: str | None) -> None:
        active = self._active_runtime()
        if identifier is None:
            return
        if active.session is not None and identifier == active.session.session_id:
            self._append_notice("该会话已经挂载在当前 pane。")
            return
        previous_id = active.session.session_id if active.session is not None else None
        mounted = self.sessions.pane_for_session(identifier)
        if mounted is not None:
            previous = self.active_pane_id
            self.sessions.active_pane_id = mounted.pane_id
            try:
                self.query_one(f"#view-{previous}").remove_class("pane-active")
                self.query_one(f"#view-{mounted.pane_id}").add_class("pane-active")
            except Exception:
                pass
            runtime = self._sync_runtime(mounted)
            self.query_one(PromptArea).focus()
            self._update_status("运行中" if runtime.busy else "就绪")
            return
        self._sync_runtime(self.sessions.switch_active(identifier))
        runtime = self._active_runtime()
        assert runtime.session is not None
        running = runtime.session.session_id in self.running_sessions
        runtime.busy = running
        if running:
            self._update_status("运行中")
            self._render_session_history(
                "已恢复仍在运行中的会话；可以继续输入消息，它们会进入队列。"
            )
            return
        self._set_pane_busy(runtime, False, "就绪")
        self._render_session_history("已恢复会话")
        if previous_id is not None and previous_id in self.running_sessions:
            self._append_notice(
                "原会话已转入后台继续运行。", runtime=runtime
            )

    def action_compact(self, instructions: str = "") -> None:
        if self.busy:
            self._append_notice("当前任务结束后才能压缩。", error=True)
            return
        self._set_busy(True, "正在压缩上下文…")
        session_id = self.session.session_id
        self.run_worker(
            lambda: self._compact_worker(instructions, session_id),
            name=f"compact-{session_id}",
            group=f"agent-{session_id}",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _compact_worker(self, instructions: str, session_id: str) -> None:
        runtime = self._runtime_for_session(session_id)
        if runtime is None:
            return
        try:
            summary = runtime.session.compact(instructions)
        except (ModelError, ValueError) as error:
            self.call_from_thread(self._finish_with_error, str(error), session_id)
            return
        self.call_from_thread(self._compact_finished, summary, session_id)

    def _compact_finished(self, summary: str, session_id: str) -> None:
        runtime = self._runtime_for_session(session_id)
        if runtime is None:
            return
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
        checkpoint = self._pending_checkpoint
        self._pending_checkpoint = None
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
        if runtime.session is None:
            return
        session = runtime.session
        for message in session.messages[1:]:
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

    def confirm_session_read(self, description: str) -> bool:
        return self._confirm_action(description, "跨会话读取确认")

    def confirm_session_control(self, description: str) -> bool:
        return self._confirm_action(description, "会话控制确认")

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

    def ask_user(
        self, session_id: str, questions: list[QuestionSpec]
    ) -> list[list[str]]:
        """Show a question dialog and wait (worker thread); reject yields None."""

        pending = _PendingQuestion(session_id, questions)
        request_id = f"q-{len(self.pending_questions) + 1}"
        self.pending_questions[request_id] = pending
        alias = self.store.session_info(session_id).alias

        def resolved(answers: list[list[str]] | None) -> None:
            pending.answers = answers
            pending.finished.set()

        self.call_from_thread(
            self.push_screen,
            QuestionPrompt(questions, f"Agent 提问 · {alias}"),
            resolved,
        )
        pending.finished.wait()
        self.pending_questions.pop(request_id, None)
        if pending.answers is None:
            raise ToolError(
                "用户取消了提问。请按你自己的判断继续执行，或改用合理的默认值。"
            )
        return pending.answers

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

    def _append_notice_card(
        self,
        title: str,
        content: str,
        runtime: PaneRuntime | None = None,
    ) -> None:
        """Long list outputs are folded into a collapsible, paged card."""

        body = PagedOutput(content)
        card = Collapsible(body, title=title, collapsed=True)
        self._mount_timeline(card, runtime)

    def _list_or_notice(
        self,
        title: str,
        content: str,
        runtime: PaneRuntime | None = None,
    ) -> None:
        """Small lists stay as plain notices; long ones become folded cards."""

        if len(content.splitlines()) > LIST_CARD_LINE_THRESHOLD:
            self._append_notice_card(title, content, runtime=runtime)
            return
        self._append_notice(f"{title}：\n{content}", runtime=runtime)

    def _append_tool(self, tool_call: ToolCall, runtime: PaneRuntime) -> None:
        body = PagedOutput("运行中…")
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
        file_change: FileChange | None = None,
    ) -> None:
        body = runtime.tool_bodies.get(tool_call.id)
        card = runtime.tool_cards.get(tool_call.id)
        if body is None or card is None:
            self._append_notice(content, error=is_error, runtime=runtime)
            return
        card.title = tool_title(tool_call, "✗" if is_error else "✓")
        card.add_class("tool-failed" if is_error else "tool-succeeded")
        if is_error:
            summary = tool_result_summary(content, is_error)
        elif file_change is not None and tool_call.name == "apply_patch":
            summary = change_result_summary(file_change)
        else:
            summary = tool_result_summary(content, is_error)
        body.set_content(summary)
        for item in self.panes.values():
            self._update_pane_header(item, "运行中" if item.busy else "就绪")

    def _finish_with_error(self, message: str, session_id: str | None = None) -> None:
        if session_id is None:
            session_id = (
                self._active_runtime().session.session_id
                if self._active_runtime().session is not None
                else None
            )
        if session_id is None:
            self._append_notice(message, error=True)
            self._set_busy(False, "错误")
            return
        self.running_sessions.discard(session_id)
        runtime = self._runtime_for_session(session_id)
        if runtime is None:
            return
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
        self._update_status(status)
        if not busy:
            self.query_one(PromptArea).focus()
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
        if runtime.session is None:
            header.update(Text(f"{runtime.pane_slot}  空窗格 · {status}"))
            return
        info = self.store.session_info(runtime.session.session_id)
        unread = len(self.store.inbox(runtime.session.session_id))
        details = f"{runtime.pane_slot}  {_session_label(info)}"
        if unread:
            details += f" · 未读 {unread}"
        if status not in {"就绪", "新 pane"}:
            details += f" · {status}"
        header.update(Text(details))

    def _update_status(self, activity: str) -> None:
        self._update_pane_header(self._active_runtime(), activity)

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
            descriptions = {
                name: item.description
                for item in COMMANDS
                for name in (item.name, *item.aliases)
            }
            candidates = list(descriptions)
            matches = _fuzzy_matches(context.query, candidates, 8)
            labels = [f"{name:<10} {descriptions[name]}" for name in matches]
        elif context.kind == "session":
            sessions = _rank_session_matches(
                self.sessions.catalog(), context.query, 12
            )
            candidates = [item.info.alias for item in sessions]
            descriptions = {
                item.info.alias: (
                    f"{_catalog_marker(item.scope, item.pane_slot)} "
                    f"{item.info.title} · {item.info.model}"
                )
                for item in sessions
            }
            matches = candidates
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
    # Mouse reporting is required for wheel scrolling and scrollbar dragging.
    # Most terminals still expose native selection via their Shift bypass.
    LitCodeTUI(settings, model).run(mouse=True)
    return 0


def _model_label(model: str, current: str) -> str:
    return f"{model}  （当前）" if model == current else model


def _session_label(info: SessionInfo) -> str:
    """Keep the collision suffix for references, not the primary UI label."""

    try:
        created = datetime.strptime(info.alias[:11], "%y%m%d-%H%M")
        timestamp = created.strftime("%m-%d %H:%M")
    except ValueError:
        timestamp = info.alias
    return f"{info.title} · {timestamp} · {info.model}"


def _catalog_marker(scope: str, pane_slot: int | None) -> str:
    if scope == "mounted" and pane_slot is not None:
        return f"[{pane_slot}]"
    if scope == "current_terminal":
        return "[本终端]"
    return "[历史]"


def _rank_session_matches(entries, query: str, limit: int):
    if not query:
        return list(entries[:limit])
    matcher = Matcher(query)
    scope_rank = {"mounted": 0, "current_terminal": 1, "history": 2}
    scored = [
        (
            scope_rank[entry.scope],
            -max(
                matcher.match(entry.info.alias),
                matcher.match(entry.info.title),
            ),
            index,
            entry,
        )
        for index, entry in enumerate(entries)
    ]
    return [
        entry
        for _, negative_score, _, entry in sorted(scored)
        if negative_score < 0
    ][:limit]


def _find_queue_message(messages, reference: str) -> str | None:
    """Accept a full queue id or its unambiguous short display prefix."""

    matches = [message.id for message in messages if message.id == reference]
    if not matches:
        matches = [message.id for message in messages if message.id.startswith(reference)]
    return matches[0] if len(matches) == 1 else None


def _history_label(
    info: SessionInfo,
    mounted: dict[str, int],
    running: set[str],
    current_session_id: str | None = None,
    depth: int = 0,
) -> Text:
    """一行有界标签：主文本截断、时间戳按节点深度贴齐同一右缘。"""

    markers = []
    if info.id == current_session_id:
        markers.append("当前")
    if info.id in mounted:
        markers.append(f"[{mounted[info.id]}]")
    elif info.id in running or info.active_turn_id is not None:
        markers.append("● 运行中")
    if info.paused:
        markers.append("已暂停")
    title = info.title if len(info.title) <= 24 else f"{info.title[:23]}…"
    main = f"{' '.join(markers)} " if markers else ""
    main += f"{info.alias} · {title}"
    if info.queue_size:
        main += f" · 队列 {info.queue_size}"
    age = _relative_time(info.updated_at)
    target = PICKER_LABEL_CELLS - depth * PICKER_GUIDE_DEPTH - _cell_width(age)
    main, _ = _truncate_cells(main, target)
    padding = target - _cell_width(main)
    label = Text(main)
    if padding > 0:
        label.append(" " * padding)
    label.append(age, style="dim")
    return label


def _truncate_cells(text: str, limit: int) -> tuple[str, bool]:
    """按显示宽度（中文占 2 列）截断文本并追加省略号。"""

    if _cell_width(text) <= limit:
        return text, False
    out: list[str] = []
    width = 0
    for character in text:
        cell = 2 if ord(character) > 0x2E80 else 1
        if width + cell > limit - 1:
            break
        out.append(character)
        width += cell
    return "".join(out) + "…", True


def _relative_time(timestamp: float) -> str:
    """Compact age for the right edge of a picker row."""

    seconds = max(0, time.time() - timestamp)
    if seconds < 90:
        return "刚刚"
    minutes = seconds / 60
    if minutes < 90:
        return f"{int(minutes)}m"
    hours = seconds / 3600
    if hours < 24:
        return f"{int(hours)}h"
    days = hours / 24
    if days < 31:
        return f"{int(days)}d"
    created = datetime.fromtimestamp(timestamp)
    return created.strftime("%m-%d")


def _cell_width(text: str) -> int:
    return sum(2 if ord(character) > 0x2E80 else 1 for character in text)


def _diff_style(line: str) -> str:
    if line.startswith("+"):
        return "bold green"
    if line.startswith("-") and not line.startswith("---"):
        return "bold red"
    if line.startswith("@"):
        return "cyan"
    if line.startswith("…"):
        return "dim"
    return ""


def _first_leaf(node: PaneNode) -> str:
    while isinstance(node, PaneBranch):
        node = node.first
    return node.pane_id


def _display_user_content(content: str) -> str:
    markers = (
        "\n\n以下是用户明确引用的本地文件快照。",
        "\n\n以下是用户明确引用的其他会话快照。",
    )
    boundaries = [content.find(marker) for marker in markers]
    visible = [boundary for boundary in boundaries if boundary >= 0]
    return content[: min(visible)] if visible else content


def _message_prompt_history(messages: list[ModelMessage]) -> list[str]:
    history: list[str] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        visible = _display_user_content(content).strip()
        if visible and (not history or history[-1] != visible):
            history.append(visible)
    return history
