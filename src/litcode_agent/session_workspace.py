"""Session and pane lifecycle orchestration, independent from Textual rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.system_prompt = system_prompt
        self.store = store
        self.event_sink = event_sink
        first = self._create("pane-1", model)
        self.panes = {first.pane_id: first}
        self.detached: dict[str, AgentSession] = {}
        self.layout = PaneLayout(first.pane_id)
        self.active_pane_id = first.pane_id
        self._next_pane_number = 2

    @property
    def active(self) -> PaneSession:
        return self.panes[self.active_pane_id]

    def split(self, direction: str) -> PaneSession:
        pane_id = f"pane-{self._next_pane_number}"
        self._next_pane_number += 1
        current = self.active
        clone = getattr(current.model, "clone_for_model", None)
        model = clone(current.model.model) if callable(clone) else current.model
        pane = self._create(pane_id, model)
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

    def _create(self, pane_id: str, model: OpenAIChatModel) -> PaneSession:
        agent = Agent(
            model,
            self.registry,
            self.settings.max_iterations,
            self.event_sink,
            HookRunner(self.settings.workspace, self.settings.hooks),
            self.system_prompt,
            self.store,
            model.model,
            self.settings.workspace,
        )
        return PaneSession(pane_id, agent, model, agent.start_session())
