"""Pane views and durable Session lifecycle.

每个 pane 始终挂载一个可寻址 Session；pane 关闭时，只有从未承载活动的
Session 才会被删除，其余 Session 转入后台。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import secrets

from litcode_agent.agent import Agent, AgentEvent, AgentSession
from litcode_agent.config import Settings
from litcode_agent.hooks import HookRunner
from litcode_agent.model import OpenAIChatModel
from litcode_agent.pane_layout import PaneLayout
from litcode_agent.scheduler import local_timezone_name
from litcode_agent.session_store import (
    Checkpoint,
    InboxMessage,
    SessionCatalogEntry,
    SessionStore,
)
from litcode_agent.session_runtime import SessionRuntime
from litcode_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class PaneSession:
    """A pane configuration and its mounted Session."""

    pane_id: str
    pane_slot: int
    agent: Agent
    model: OpenAIChatModel
    session: AgentSession


class SessionWorkspace:
    """Own pane topology and every AgentSession lifecycle transition."""

    def __init__(
        self,
        settings: Settings,
        model: OpenAIChatModel,
        registry: ToolRegistry,
        system_prompt: str,
        store: SessionStore,
        event_sink: Callable[[AgentEvent], None],
        terminal_id: str | None = None,
        runtime: SessionRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.system_prompt = system_prompt
        self.store = store
        self.event_sink = event_sink
        self.terminal_id = terminal_id or _terminal_id()
        self.runtime = runtime
        self.default_model = model
        first = self._create(1, model)
        self.panes = {first.pane_id: first}
        self.detached: dict[str, AgentSession] = {}
        self.layout = PaneLayout(first.pane_id)
        self.active_pane_id = first.pane_id

    def switch_provider(self, model: OpenAIChatModel) -> None:
        """Replace the endpoint for every mounted pane and recorded session.

        /connect 在供货商之间切换；各 pane 统一改用新模型 ID，历史消息保留。
        """

        self.default_model = model
        for pane in self.panes.values():
            pane.model = _clone_model(model, model.model)
            pane.agent.model = pane.model
            pane.agent.model_name = model.model
            self.store.update_model(pane.session.session_id, model.model)

    @property
    def active(self) -> PaneSession:
        return self.panes[self.active_pane_id]

    def split(
        self, direction: str, session_id: str | None = None
    ) -> PaneSession:
        pane_slot = next(
            slot for slot in range(1, 5) if f"pane-{slot}" not in self.panes
        )
        pane_id = f"pane-{pane_slot}"
        current = self.active
        model = _clone_model(
            current.model,
            self.store.session_info(session_id).model
            if session_id is not None
            else current.model.model,
        )
        pane = self._create(pane_slot, model)
        if session_id is not None:
            self._release_session(pane.session)
            live = self.runtime.session_for(session_id) if self.runtime else None
            detached = self.detached.pop(session_id, None)
            if live is not None:
                pane.session = live
            elif detached is not None:
                pane.session = detached
                pane.agent = detached.agent
                pane.model = _model_from_agent(detached.agent, model)
                self._bind_runtime_location(pane)
            else:
                pane.session = pane.agent.start_session(session_id)
            if self.runtime is not None:
                self.runtime.register(session_id, pane.session)
        self.layout.split(self.active_pane_id, direction, pane_id)
        self.panes[pane_id] = pane
        self.active_pane_id = pane_id
        return pane

    def focus(self, direction: str) -> PaneSession | None:
        target = self.layout.focus_from(self.active_pane_id, direction)
        if target is None:
            return None
        self.active_pane_id = target
        return self.active

    def focus_next(self) -> PaneSession:
        pane_ids = self.layout.pane_ids()
        index = pane_ids.index(self.active_pane_id)
        self.active_pane_id = pane_ids[(index + 1) % len(pane_ids)]
        return self.active

    def close_active_pane(self) -> tuple[str, PaneSession]:
        if len(self.panes) == 1:
            raise ValueError("cannot close the final pane")
        removed = self.active
        pane_ids = self.layout.pane_ids()
        index = pane_ids.index(removed.pane_id)
        fallback = pane_ids[index - 1] if index else pane_ids[1]
        self._release_session(removed.session)
        self.layout.close(removed.pane_id)
        del self.panes[removed.pane_id]
        self.active_pane_id = fallback
        return removed.pane_id, self.active

    def detach_active(self) -> AgentSession:
        """Move the active Session to background and mount a new root Session."""

        pane = self.active
        session = pane.session
        self.detached[session.session_id] = session
        pane.session = pane.agent.start_session()
        if self.runtime is not None:
            self.runtime.register(pane.session.session_id, pane.session)
        return session

    def clear_active(self) -> AgentSession:
        pane = self.active
        current = pane.session
        if not self._delete_if_pristine(current):
            current.close("user_clear", "")
        pane.session = pane.agent.start_session()
        if self.runtime is not None:
            self.runtime.register(pane.session.session_id, pane.session)
        return pane.session

    def new_active(self) -> AgentSession:
        pane = self.active
        self._release_session(pane.session)
        pane.session = pane.agent.start_session()
        if self.runtime is not None:
            self.runtime.register(pane.session.session_id, pane.session)
        return pane.session

    def switch_active(self, session_id: str) -> PaneSession:
        pane = self.active
        if session_id == pane.session.session_id:
            return pane
        self._release_session(pane.session)
        detached = self.detached.pop(session_id, None)
        if detached is not None:
            pane.session = detached
            pane.agent = detached.agent
            pane.model = _model_from_agent(detached.agent, pane.model)
            self._bind_runtime_location(pane)
            if self.runtime is not None:
                self.runtime.register(session_id, pane.session)
            return pane
        info = self.store.session_info(session_id)
        pane.model = _clone_model(pane.model, info.model)
        pane.agent.model = pane.model
        pane.agent.model_name = info.model
        live = self.runtime.session_for(session_id) if self.runtime else None
        pane.session = live or pane.agent.start_session(session_id)
        if self.runtime is not None:
            self.runtime.register(session_id, pane.session)
        return pane

    def fork_active(self, checkpoint: Checkpoint) -> AgentSession:
        pane = self.active
        current = pane.session
        current.close("user_fork", "")
        pane.session = current.fork(checkpoint)
        if self.runtime is not None:
            self.runtime.register(pane.session.session_id, pane.session)
        return pane.session

    def select_model(self, model_name: str) -> None:
        pane = self.active
        pane.model.select_model(model_name)
        pane.model = _clone_model(pane.model, model_name)
        pane.agent.model = pane.model
        pane.agent.model_name = model_name
        self.store.update_model(pane.session.session_id, model_name)

    def consume_inbox(self) -> tuple[InboxMessage, ...]:
        """Return unread messages and advance their visible state."""

        session_id = self.active.session.session_id
        messages = self.store.inbox(session_id)
        for message in messages:
            self.store.mark_inbox_read(session_id, message.id)
        return messages

    def close_all(self) -> None:
        closed: set[str] = set()
        for pane in self.panes.values():
            if not self._delete_if_pristine(pane.session):
                pane.session.close("user_exit", "")
            closed.add(pane.session.session_id)
        for session_id, session in self.detached.items():
            if session_id not in closed:
                session.close("user_exit", "")

    def _create(
        self,
        pane_slot: int,
        model: OpenAIChatModel,
        *,
        session_id: str | None = None,
    ) -> PaneSession:
        pane_id = f"pane-{pane_slot}"
        agent = Agent(
            model=model,
            tools=self.registry,
            max_iterations=self.settings.max_iterations,
            event_sink=self.event_sink,
            hooks=HookRunner(self.settings.workspace, self.settings.hooks),
            system_prompt=self.system_prompt,
            store=self.store,
            model_name=model.model,
            workspace=self.settings.workspace,
            runtime_context=lambda: self._runtime_location(pane_id),
            tool_context=lambda current_id: self._tool_context(pane_id, current_id),
            origin_terminal_id=self.terminal_id,
            origin_pane_slot=pane_slot,
            auto_compact_chars=self.settings.auto_compact_chars,
        )
        session = agent.start_session(session_id)
        if self.runtime is not None:
            self.runtime.register(session.session_id, session)
        return PaneSession(pane_id, pane_slot, agent, model, session)

    def _release_session(self, session: AgentSession) -> None:
        """Delete an unused root Session, otherwise keep it alive in background."""

        if self._delete_if_pristine(session):
            return
        self.detached[session.session_id] = session

    def _delete_if_pristine(self, session: AgentSession) -> bool:
        if not self.store.delete_if_pristine(session.session_id):
            return False
        if self.runtime is not None:
            self.runtime.unregister(session.session_id)
        session.close("unused_pane_closed", "")
        return True

    def _bind_runtime_location(self, pane: PaneSession) -> None:
        pane.agent.runtime_context = lambda: self._runtime_location(pane.pane_id)
        pane.agent.tool_context = lambda current_id: self._tool_context(
            pane.pane_id, current_id
        )
        pane.agent.origin_terminal_id = self.terminal_id
        pane.agent.origin_pane_slot = pane.pane_slot

    def mounted_sessions(self) -> dict[str, int]:
        return {
            pane.session.session_id: pane.pane_slot
            for pane in self.panes.values()
        }

    def pane_for_session(self, session_id: str) -> PaneSession | None:
        return next(
            (
                pane
                for pane in self.panes.values()
                if pane.session.session_id == session_id
            ),
            None,
        )

    def catalog(self, query: str = "", limit: int = 50):
        return self.store.session_catalog(
            self.settings.workspace,
            current_terminal_id=self.terminal_id,
            mounted=self.mounted_sessions(),
            query=query,
            limit=limit,
        )

    def _tool_context(self, pane_id: str | None, session_id: str):
        from litcode_agent.tools.base import ToolExecutionContext

        pane = self.panes.get(pane_id) if pane_id is not None else None
        info = self.store.session_info(session_id)
        return ToolExecutionContext(
            session_id,
            self.settings.workspace.resolve(),
            terminal_id=self.terminal_id,
            pane_slot=pane.pane_slot if pane is not None else None,
            mounted_sessions=tuple(sorted(self.mounted_sessions().items())),
            profile=info.profile,
            turn_id=info.active_turn_id,
            runtime=self.runtime,
        )

    def _runtime_location(self, pane_id: str) -> str:
        pane = self.panes[pane_id]
        current = self.store.session_info(pane.session.session_id)
        mounted = []
        for candidate in sorted(self.panes.values(), key=lambda item: item.pane_slot):
            info = self.store.session_info(candidate.session.session_id)
            mounted.append(f"{candidate.pane_slot}={info.alias} · {info.title}")
        return "\n".join(
            (
                self._time_context(),
                f"当前终端：{self.terminal_id}",
                f"当前 pane：{pane.pane_slot}",
                f"当前会话：{current.alias}",
                "同终端 pane：" + "；".join(mounted),
                *(
                    (sibling,)
                    if (sibling := self.sibling_sessions_section(
                        pane.session.session_id
                    ))
                    else ()
                ),
                "pane 编号只在本次 TUI 运行中有效。",
            )
        )

    def _runtime_location_actor(self, session_id: str) -> str:
        """Runtime location for a background/child actor without a Pane."""

        info = self.store.session_info(session_id)
        return "\n".join(
            (
                self._time_context(),
                f"当前终端：{self.terminal_id}",
                "当前 pane：无（后台子会话）",
                f"当前会话：{info.alias}",
                *(
                    (sibling,)
                    if (sibling := self.sibling_sessions_section(session_id))
                    else ()
                ),
            )
        )

    def _time_context(self) -> str:
        return (
            f"当前本地时间：{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"默认 IANA 时区：{local_timezone_name()}"
        )

    def sibling_sessions_section(self, current_id: str) -> str:
        """Bounded snapshot of other workspace sessions, for model context."""

        entries = self.store.session_catalog(
            self.settings.workspace,
            current_terminal_id=self.terminal_id,
            mounted=self.mounted_sessions(),
            limit=10,
        )
        return format_sibling_sessions(entries, current_id)

    def session_factory(self, session_id: str, profile: str) -> AgentSession:
        """Build an AgentSession for a background/child actor without a Pane."""

        info = self.store.session_info(session_id)
        model = _clone_model(self.active.model, info.model)
        suffix = (
            "\n\n你是只读调查 Agent；不得修改工作区文件或运行可能写入工作区的命令。"
            if profile == "explore"
            else ""
        )
        agent = Agent(
            model=model,
            tools=self.registry,
            max_iterations=self.settings.max_iterations,
            event_sink=self.event_sink,
            hooks=HookRunner(self.settings.workspace, self.settings.hooks),
            system_prompt=self.system_prompt + suffix,
            store=self.store,
            model_name=info.model,
            workspace=self.settings.workspace,
            runtime_context=lambda: self._runtime_location_actor(session_id),
            tool_context=lambda current_id: self._tool_context(None, current_id),
            origin_terminal_id=self.terminal_id,
            origin_pane_slot=None,
            auto_compact_chars=self.settings.auto_compact_chars,
        )
        return agent.start_session(session_id)


def _clone_model(model: OpenAIChatModel, model_name: str) -> OpenAIChatModel:
    """Clone provider clients when available; test doubles can be reused."""

    if isinstance(model, OpenAIChatModel):
        return model.clone_for_model(model_name)
    return model


def _model_from_agent(agent: Agent, fallback: OpenAIChatModel) -> OpenAIChatModel:
    return agent.model if isinstance(agent.model, OpenAIChatModel) else fallback


def _terminal_id() -> str:
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    return "T-" + "".join(secrets.choice(alphabet) for _ in range(3))


SIBLING_GUIDE = """协作工具（只读有界）：list_sessions 刷新本快照并设置查询；read_session_context 按查询读取对方会话的匹配片段；send_session_message 向对方 inbox 投递指令（不唤醒、不打断对方运行）；read_session_inbox 读取自己收到的投递。
与这些会话协作时优先只读；不要重复对方正在做的工作；若任务需要对方同步，向对方投递明确分工，不确定就先查询再决定。"""


def format_sibling_sessions(
    entries: tuple[SessionCatalogEntry, ...], current_id: str
) -> str:
    """Render a bounded liveness snapshot of sibling sessions.

    Only metadata leaves the session store; messages stay private until the
    model deliberately queries a peer (read_session_context)."""

    visible = [entry for entry in entries if entry.info.id != current_id]
    if not visible:
        return ""
    lines = [
        "<litcode_sibling_sessions>",
        "同工作区其他会话快照（有界元数据；下面的位置号只在本次 TUI 运行中有效）：",
    ]
    for entry in visible:
        info = entry.info
        if entry.scope == "mounted":
            where = f"pane {entry.pane_slot}"
        elif entry.scope == "current_terminal":
            where = "同终端后台"
        else:
            where = "后台"
        status = (info.status or "idle").strip()
        activity = (info.activity or "").strip()
        queue = f" · 队列 {info.queue_size}" if info.queue_size else ""
        line = f"- {info.alias} · {where} · {status or 'idle'}"
        if activity:
            line += f" · {activity}"
        lines.append(line + queue)
    lines.append(SIBLING_GUIDE)
    lines.append("</litcode_sibling_sessions>")
    return "\n".join(lines)
