"""Pane views and durable Session lifecycle.

一个 pane 可以暂时没有会话。Session 是持久化的执行主体，pane 只是它的
可见挂载点；因此卸载、重新挂载和后台运行都不需要伪造一个空会话。
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
from litcode_agent.session_store import Checkpoint, InboxMessage, SessionStore
from litcode_agent.session_runtime import SessionRuntime
from litcode_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class PaneSession:
    """A pane configuration and its optional mounted Session."""

    pane_id: str
    pane_slot: int
    agent: Agent
    model: OpenAIChatModel
    session: AgentSession | None

    @property
    def empty(self) -> bool:
        return self.session is None


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
            if pane.session is not None:
                self.store.update_model(pane.session.session_id, model.model)

    @property
    def active(self) -> PaneSession:
        return self.panes[self.active_pane_id]

    def materialize(self, pane_id: str | None = None) -> AgentSession:
        """Create a root Session for an Empty Pane on its first real use."""

        pane = self.panes[pane_id or self.active_pane_id]
        if pane.session is None:
            pane.session = pane.agent.start_session()
            if self.runtime is not None:
                self.runtime.register(pane.session.session_id, pane.session)
        return pane.session

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
            if self.runtime is not None and pane.session is not None:
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
        if removed.session is not None:
            self.detached[removed.session.session_id] = removed.session
        self.layout.close(removed.pane_id)
        del self.panes[removed.pane_id]
        self.active_pane_id = fallback
        return removed.pane_id, self.active

    def detach_active(self) -> AgentSession:
        """Unmount the active Session while keeping its actor alive."""

        pane = self.active
        if pane.session is None:
            raise ValueError("the active pane is already empty")
        session = pane.session
        self.detached[session.session_id] = session
        pane.session = None
        return session

    def clear_active(self) -> AgentSession:
        pane = self.active
        current = pane.session or self.materialize(pane.pane_id)
        current.close("user_clear", "")
        pane.session = pane.agent.start_session()
        if self.runtime is not None:
            self.runtime.register(pane.session.session_id, pane.session)
        return pane.session

    def new_active(self) -> AgentSession:
        pane = self.active
        if pane.session is not None:
            self.detached[pane.session.session_id] = pane.session
        pane.session = pane.agent.start_session()
        if self.runtime is not None:
            self.runtime.register(pane.session.session_id, pane.session)
        return pane.session

    def switch_active(self, session_id: str) -> PaneSession:
        pane = self.active
        if pane.session is not None and session_id == pane.session.session_id:
            return pane
        if pane.session is not None:
            self.detached[pane.session.session_id] = pane.session
            pane.session = None
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
        current = pane.session or self.materialize(pane.pane_id)
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
        if pane.session is not None:
            self.store.update_model(pane.session.session_id, model_name)

    def consume_inbox(self) -> tuple[InboxMessage, ...]:
        """Return unread messages and advance their visible state."""

        if self.active.session is None:
            return ()
        session_id = self.active.session.session_id
        messages = self.store.inbox(session_id)
        for message in messages:
            self.store.mark_inbox_read(session_id, message.id)
        return messages

    def close_all(self) -> None:
        closed: set[str] = set()
        for pane in self.panes.values():
            if pane.session is not None:
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
        )
        session = agent.start_session(session_id) if session_id is not None else None
        return PaneSession(pane_id, pane_slot, agent, model, session)

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
            if pane.session is not None
        }

    def pane_for_session(self, session_id: str) -> PaneSession | None:
        return next(
            (
                pane
                for pane in self.panes.values()
                if pane.session is not None and pane.session.session_id == session_id
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
        time_context = (
            f"当前本地时间：{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"默认 IANA 时区：{local_timezone_name()}"
        )
        if pane.session is None:
            return (
                f"{time_context}\n"
                f"当前终端：{self.terminal_id}\n"
                f"当前 pane：{pane.pane_slot}\n"
                "当前 pane 为空；首次输入后创建会话。"
            )
        current = self.store.session_info(pane.session.session_id)
        mounted = []
        for candidate in sorted(self.panes.values(), key=lambda item: item.pane_slot):
            if candidate.session is None:
                mounted.append(f"{candidate.pane_slot}=空窗格")
                continue
            info = self.store.session_info(candidate.session.session_id)
            mounted.append(f"{candidate.pane_slot}={info.alias} · {info.title}")
        return "\n".join(
            (
                time_context,
                f"当前终端：{self.terminal_id}",
                f"当前 pane：{pane.pane_slot}",
                f"当前会话：{current.alias}",
                "同终端 pane：" + "；".join(mounted),
                "pane 编号只在本次 TUI 运行中有效。",
            )
        )

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
            runtime_context=None,
            tool_context=lambda current_id: self._tool_context(None, current_id),
            origin_terminal_id=self.terminal_id,
            origin_pane_slot=None,
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
