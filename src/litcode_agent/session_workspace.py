"""Session and pane lifecycle orchestration, independent from Textual rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import secrets

from litcode_agent.agent import Agent, AgentEvent, AgentSession
from litcode_agent.config import Settings
from litcode_agent.hooks import HookRunner
from litcode_agent.model import OpenAIChatModel
from litcode_agent.pane_layout import PaneLayout
from litcode_agent.session_store import Checkpoint, InboxMessage, SessionStore
from litcode_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class PaneSession:
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
        before_model_request: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.system_prompt = system_prompt
        self.store = store
        self.event_sink = event_sink
        self.terminal_id = terminal_id or _terminal_id()
        self.before_model_request = before_model_request
        first = self._create(1, model)
        self.panes = {first.pane_id: first}
        self.detached: dict[str, AgentSession] = {}
        self.layout = PaneLayout(first.pane_id)
        self.active_pane_id = first.pane_id

    @property
    def active(self) -> PaneSession:
        return self.panes[self.active_pane_id]

    def split(self, direction: str) -> PaneSession:
        pane_slot = next(
            slot
            for slot in range(1, 5)
            if f"pane-{slot}" not in self.panes
        )
        pane_id = f"pane-{pane_slot}"
        current = self.active
        clone = getattr(current.model, "clone_for_model", None)
        model = clone(current.model.model) if callable(clone) else current.model
        pane = self._create(pane_slot, model)
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
        self.detached[removed.session.session_id] = removed.session
        self.layout.close(removed.pane_id)
        del self.panes[removed.pane_id]
        self.active_pane_id = fallback
        return removed.pane_id, self.active

    def clear_active(self) -> AgentSession:
        pane = self.active
        pane.session.close("user_clear", "")
        pane.session = pane.agent.start_session()
        return pane.session

    def switch_active(self, session_id: str) -> PaneSession:
        pane = self.active
        if session_id == pane.session.session_id:
            return pane
        pane.session.close("user_switch", "")
        detached = self.detached.pop(session_id, None)
        if detached is not None:
            pane.session = detached
            pane.agent = detached.agent
            pane.model = detached.agent.model  # type: ignore[assignment]
            self._bind_runtime_location(pane)
            return pane
        info = self.store.session_info(session_id)
        pane.model.select_model(info.model)
        pane.agent.model_name = info.model
        pane.session = pane.agent.start_session(session_id)
        return pane

    def fork_active(self, checkpoint: Checkpoint) -> AgentSession:
        pane = self.active
        pane.session.close("user_fork", "")
        pane.session = pane.session.fork(checkpoint)
        return pane.session

    def select_model(self, model_name: str) -> None:
        pane = self.active
        pane.model.select_model(model_name)
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
        for pane in self.panes.values():
            pane.session.close("user_exit", "")
        for session in self.detached.values():
            session.close("user_exit", "")

    def _create(self, pane_slot: int, model: OpenAIChatModel) -> PaneSession:
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
            tool_context=lambda session_id: self._tool_context(pane_id, session_id),
            origin_terminal_id=self.terminal_id,
            origin_pane_slot=pane_slot,
            before_model_request=self.before_model_request,
        )
        return PaneSession(pane_id, pane_slot, agent, model, agent.start_session())

    def _bind_runtime_location(self, pane: PaneSession) -> None:
        pane.agent.runtime_context = lambda: self._runtime_location(pane.pane_id)
        pane.agent.tool_context = lambda session_id: self._tool_context(
            pane.pane_id, session_id
        )
        pane.agent.origin_terminal_id = self.terminal_id
        pane.agent.origin_pane_slot = pane.pane_slot

    def mounted_sessions(self) -> dict[str, int]:
        return {
            pane.session.session_id: pane.pane_slot for pane in self.panes.values()
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

    def _tool_context(self, pane_id: str, session_id: str):
        from litcode_agent.tools.base import ToolExecutionContext

        running_task = self.store.running_task_for_session(session_id)
        role = running_task.role if running_task is not None else None
        write_policy = (
            running_task.write_policy if running_task is not None else None
        )
        allowed_paths = (
            running_task.allowed_paths if running_task is not None else ()
        )
        if running_task is None:
            run = self.store.active_orchestration_run(
                self.settings.workspace, session_id
            )
            if run is not None and run.status == "running":
                role = "coordinator"
                write_policy = "none"
        return ToolExecutionContext(
            session_id,
            self.settings.workspace.resolve(),
            terminal_id=self.terminal_id,
            pane_slot=self.panes[pane_id].pane_slot,
            mounted_sessions=tuple(sorted(self.mounted_sessions().items())),
            orchestration_role=role,
            orchestration_write_policy=write_policy,
            orchestration_allowed_paths=allowed_paths,
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
                f"当前终端：{self.terminal_id}",
                f"当前 pane：{pane.pane_slot}",
                f"当前会话：{current.alias}",
                "同终端 pane：" + "；".join(mounted),
                "pane 编号只在本次 TUI 运行中有效。",
            )
        )


def _terminal_id() -> str:
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    return "T-" + "".join(secrets.choice(alphabet) for _ in range(3))
