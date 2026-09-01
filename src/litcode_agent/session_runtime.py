"""Lightweight in-process Session actors.

The runtime intentionally has only three moving parts:

* SQLite stores the Session mailbox and durable turn state;
* one small worker loop consumes one Session mailbox at a time;
* a fair process-wide slot gate limits concurrent model/tool work.

There is no second task protocol here.  A child Agent is just a
Session with ``parent_id`` and a normal Agent turn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from io import TextIOWrapper
import threading
import time
import uuid
import sqlite3
import os
from pathlib import Path
import fcntl
from typing import Literal

from litcode_agent.agent import AgentResult, AgentSession
from litcode_agent.session_store import (
    QueuedMessage,
    SessionInfo,
    SessionStore,
    SessionTurn,
)


class SessionRuntimeError(RuntimeError):
    """A visible Session actor, budget, or lifecycle violation."""


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Stable capability defaults; task-specific behaviour belongs in prompt."""

    name: str
    system_prompt: str = ""
    allow_writes: bool = True
    allow_commands: bool = True


GENERAL_PROFILE = AgentProfile("general")
EXPLORE_PROFILE = AgentProfile(
    "explore",
    "你是只读调查 Agent；不得修改工作区文件，也不得运行可能写入工作区的命令。",
    allow_writes=False,
    allow_commands=False,
)
PROFILES: dict[str, AgentProfile] = {
    GENERAL_PROFILE.name: GENERAL_PROFILE,
    EXPLORE_PROFILE.name: EXPLORE_PROFILE,
}

InvocationStatus = Literal[
    "queued", "running", "waiting", "completed", "failed", "cancelled"
]


@dataclass(frozen=True, slots=True)
class SubagentInvocation:
    id: str
    parent_session_id: str
    child_session_id: str
    parent_turn_id: str
    root_turn_id: str
    background: bool
    status: InvocationStatus
    prompt: str
    created_at: float
    output: str | None = None


@dataclass(frozen=True, slots=True)
class SpawnResult:
    invocation_id: str
    child_session_id: str
    child_alias: str
    background: bool
    output: str | None = None


@dataclass(slots=True)
class _Actor:
    session_id: str
    session: AgentSession | None = None
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )
    thread: threading.Thread | None = None
    stop: bool = False
    cancel_event: threading.Event | None = None


SessionFactory = Callable[[str, str], AgentSession]
EventSink = Callable[[str, str, SessionTurn | None], None]


class SessionRuntime:
    """Run durable Session mailboxes as single-consumer actors."""

    def __init__(
        self,
        store: SessionStore,
        workspace: Path,
        session_factory: SessionFactory | None = None,
        *,
        system_prompt: str = "You are LitCode Agent.",
        max_depth: int = 2,
        invocation_budget: int = 8,
        max_slots: int = 4,
        event_sink: EventSink | None = None,
        claim_workspace: bool = True,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must not be negative")
        if invocation_budget <= 0:
            raise ValueError("invocation_budget must be positive")
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self.store = store
        self.workspace = workspace.resolve()
        self.session_factory = session_factory
        self.system_prompt = system_prompt
        self.max_depth = max_depth
        self.invocation_budget = invocation_budget
        self.max_slots = max_slots
        self.event_sink = event_sink or (lambda kind, session_id, turn: None)

        self._actors: dict[str, _Actor] = {}
        self._actors_lock = threading.RLock()
        self._stopping = False
        self._slot_condition = threading.Condition(threading.RLock())
        self._slots_in_use = 0
        self._slot_waiters: list[str] = []
        self._local = threading.local()
        self._invocations: dict[str, SubagentInvocation] = {}
        self._invocation_cancel: dict[str, threading.Event] = {}
        self._invocation_messages: dict[str, str] = {}
        self._invocations_lock = threading.RLock()
        self._children: dict[str, set[str]] = {}
        self._budgets: dict[str, int] = {}
        self._workspace_lock_file = None
        if claim_workspace:
            self._workspace_lock_file = self._claim_workspace(self.workspace)

        # An application restart interrupts active work, but does not erase
        # history or queued messages.  The caller may explicitly resume them.
        self.store.recover_session_runtime(self.workspace)

    # ------------------------------------------------------------------
    # Actor lifecycle and mailbox operations

    def register(self, session_id: str, session: AgentSession) -> None:
        """Attach an already configured AgentSession to a durable actor."""

        if not self.store.session_belongs_to(self.workspace, session_id):
            raise SessionRuntimeError(f"session is outside workspace: {session_id}")
        with self._actors_lock:
            actor = self._actors.setdefault(session_id, _Actor(session_id))
            actor.session = session
        self._wake(session_id)

    def submit(
        self,
        session_id: str,
        content: str,
        *,
        source_session_id: str | None = None,
        kind: str = "message",
    ) -> QueuedMessage:
        if self._stopping:
            raise SessionRuntimeError("runtime is stopping")
        message = self.store.enqueue_message(
            session_id,
            content,
            source_session_id=source_session_id,
            workspace=self.workspace,
            kind=kind,
        )
        self._ensure_actor(session_id)
        self._wake(session_id)
        return message

    def send_message(
        self,
        source_session_id: str,
        target_session_id: str,
        content: str,
        *,
        kind: str = "message",
    ) -> QueuedMessage:
        """Append to the target's one shared queue without interrupting it."""

        return self.submit(
            target_session_id,
            content,
            source_session_id=source_session_id,
            kind=kind,
        )

    def create_subagent_session(
        self,
        parent_session_id: str,
        prompt: str,
        *,
        profile: str = "general",
        start: bool = True,
    ) -> SessionInfo:
        """Create a user-requested child Session outside an Agent turn."""

        prompt = prompt.strip()
        if not prompt:
            raise SessionRuntimeError("subagent prompt must not be empty")
        self._ensure_workspace_session(parent_session_id)
        profile_info = PROFILES.get(profile)
        if profile_info is None:
            raise SessionRuntimeError(f"unknown agent profile: {profile}")
        if self.store.session_depth(parent_session_id) >= self.max_depth:
            raise SessionRuntimeError(
                f"maximum session depth reached ({self.max_depth})"
            )
        parent = self.store.session_info(parent_session_id)
        if parent.profile == "explore" and profile != "explore":
            raise SessionRuntimeError(
                "explore profile cannot delegate a more capable profile"
            )
        child_id = self.store.create_child(
            parent_session_id,
            parent.model,
            [{"role": "system", "content": self._child_system_prompt(profile_info)}],
            title=prompt.replace("\n", " ")[:48] or "子会话",
            profile=profile,
        )
        if start:
            self.submit(
                child_id,
                prompt,
                source_session_id=parent_session_id,
                kind="user_subagent",
            )
        return self.store.session_info(child_id)

    def pause(self, session_id: str) -> SessionInfo:
        self._ensure_workspace_session(session_id)
        info = self.store.set_paused(session_id, True)
        self.event_sink("paused", session_id, None)
        return info

    def resume(self, session_id: str) -> SessionInfo:
        self._ensure_workspace_session(session_id)
        info = self.store.set_paused(session_id, False)
        self._ensure_actor(session_id)
        self._wake(session_id)
        self.event_sink("resumed", session_id, None)
        return info

    def session_for(self, session_id: str) -> AgentSession | None:
        """Return the live AgentSession when a background actor has one."""

        with self._actors_lock:
            actor = self._actors.get(session_id)
            return actor.session if actor is not None else None

    def running(self, session_id: str) -> bool:
        info = self.store.session_info(session_id)
        return info.status in {"waiting", "running"}

    def wait_for_idle(self, session_id: str, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._session_quiet(session_id):
                return True
            time.sleep(0.01)
        return False

    def _session_quiet(self, session_id: str) -> bool:
        info = self.store.session_info(session_id)
        queued = self.store.queue(session_id)
        return info.status not in {"waiting", "running"} and not any(
            message.status in {"queued", "running"} for message in queued
        )

    # ------------------------------------------------------------------
    # Child creation and collaboration

    def spawn_subagent(
        self,
        parent_session_id: str,
        prompt: str,
        *,
        agent: str | None = None,
        background: bool = False,
        session_id: str | None = None,
    ) -> SpawnResult:
        prompt = prompt.strip()
        if not prompt:
            raise SessionRuntimeError("subagent prompt must not be empty")
        self._ensure_workspace_session(parent_session_id)
        current_turn = self._current_turn()
        if current_turn is None or current_turn.session_id != parent_session_id:
            raise SessionRuntimeError("spawn_subagent requires the active parent turn")
        parent_info = self.store.session_info(parent_session_id)
        parent_profile = PROFILES.get(parent_info.profile, GENERAL_PROFILE)
        if parent_profile.name == "explore" and agent not in {None, "explore"}:
            raise SessionRuntimeError(
                "explore profile cannot delegate a more capable profile"
            )
        root_turn_id = current_turn.root_turn_id
        count = self._budgets.get(root_turn_id, 0)
        if count >= self.invocation_budget:
            raise SessionRuntimeError(
                f"subagent invocation budget exhausted ({self.invocation_budget})"
            )
        if self.store.session_depth(parent_session_id) >= self.max_depth:
            raise SessionRuntimeError(
                f"maximum session depth reached ({self.max_depth})"
            )

        if session_id is not None:
            self._ensure_workspace_session(session_id)
            child_info = self.store.session_info(session_id)
            if child_info.parent_id != parent_session_id:
                raise SessionRuntimeError(
                    "session_id must identify a child created directly by this session"
                )
            profile = PROFILES.get(agent or child_info.profile)
            if profile is None:
                raise SessionRuntimeError(f"unknown agent profile: {agent}")
            if agent is not None and child_info.profile != profile.name:
                raise SessionRuntimeError("continued child profile does not match")
            child_id = session_id
        else:
            profile = PROFILES.get(agent or parent_profile.name)
            if profile is None:
                raise SessionRuntimeError(f"unknown agent profile: {agent}")
            child_id = self.store.create_child(
                parent_session_id,
                parent_info.model,
                [{"role": "system", "content": self._child_system_prompt(profile)},],
                title=prompt.replace("\n", " ")[:48] or "子会话",
                profile=profile.name,
            )

        invocation_id = str(uuid.uuid4())
        invocation = SubagentInvocation(
            id=invocation_id,
            parent_session_id=parent_session_id,
            child_session_id=child_id,
            parent_turn_id=current_turn.id,
            root_turn_id=root_turn_id,
            background=background,
            status="queued",
            prompt=prompt,
            created_at=time.time(),
        )
        with self._invocations_lock:
            self._invocations[invocation_id] = invocation
            self._invocation_cancel[invocation_id] = threading.Event()
            self._children.setdefault(current_turn.id, set()).add(invocation_id)
            self._budgets[root_turn_id] = count + 1

        try:
            message = self.submit(
                child_id,
                prompt,
                source_session_id=parent_session_id,
                kind="subagent_invocation",
            )
        except Exception:
            # A failed enqueue must not consume a durable-looking invocation
            # slot or leave an orphan child in the in-memory ownership map.
            with self._invocations_lock:
                self._invocations.pop(invocation_id, None)
                self._invocation_cancel.pop(invocation_id, None)
                self._invocation_messages.pop(invocation_id, None)
                self._children.get(current_turn.id, set()).discard(invocation_id)
                self._budgets[root_turn_id] = count
            raise
        with self._invocations_lock:
            self._invocation_messages[invocation_id] = message.id
        if background:
            return SpawnResult(
                invocation_id,
                child_id,
                self.store.session_info(child_id).alias,
                True,
            )

        # Waiting for a foreground child is not active work: release the
        # parent's slot to avoid a parent/child deadlock at max_slots=1.
        self._update_invocation(invocation_id, status="waiting")
        self._release_current_slot(current_turn.id)
        try:
            output = self._wait_for_invocation(invocation_id)
        finally:
            if not self._invocation_cancel[invocation_id].is_set():
                if self._acquire_slot(current_turn.id, self.current_cancel_event() or threading.Event()):
                    self._local.slot_turn_id = current_turn.id
        return SpawnResult(
            invocation_id,
            child_id,
            self.store.session_info(child_id).alias,
            False,
            output,
        )

    def wait_for_session(self, target: str, timeout: float = 600.0) -> str:
        """Block until a Session or background invocation is quiet.

        Like a user confirmation this performs no active work: the current
        turn's execution slot is released for the duration of the wait so a
        foreground parent never deadlocks a child at max_slots=1.
        """

        turn = self._current_turn()
        if turn is None:
            raise SessionRuntimeError("wait_for_session requires the active parent turn")
        target_id, invocation = self._resolve_wait_target(target)
        if target_id == turn.session_id:
            raise SessionRuntimeError("a session cannot wait for itself")
        cancel_event = self.current_cancel_event() or threading.Event()
        released = self._release_current_slot(turn.id)
        if not released:
            return self._wait_until_quiet(target_id, invocation, timeout, cancel_event)
        self.store.update_turn(turn.id, status="waiting")
        self.store.set_session_state(
            turn.session_id,
            status="waiting",
            activity="等待子会话完成",
            active_turn_id=turn.id,
        )
        try:
            if cancel_event.is_set() or self._stopping:
                raise SessionRuntimeError("wait_for_session was cancelled")
            return self._wait_until_quiet(target_id, invocation, timeout, cancel_event)
        finally:
            reacquired = self._acquire_slot(turn.id, cancel_event)
            if reacquired:
                self._local.slot_turn_id = turn.id
                self.store.update_turn(turn.id, status="running")
                self.store.set_session_state(
                    turn.session_id,
                    status="running",
                    activity="正在执行",
                    active_turn_id=turn.id,
                )

    def _resolve_wait_target(
        self, target: str
    ) -> tuple[str, SubagentInvocation | None]:
        with self._invocations_lock:
            for invocation in self._invocations.values():
                if invocation.id == target:
                    return invocation.child_session_id, invocation
        try:
            target_id = self.store.session_id_for_reference(self.workspace, target)
        except KeyError as error:
            raise SessionRuntimeError(f"找不到会话（当前工作区）：{target}") from error
        return target_id, None

    def _wait_until_quiet(
        self,
        target_id: str,
        invocation: SubagentInvocation | None,
        timeout: float,
        cancel_event: threading.Event,
    ) -> str:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if cancel_event.is_set() or self._stopping:
                raise SessionRuntimeError("wait_for_session was cancelled")
            if time.monotonic() >= deadline:
                raise SessionRuntimeError(
                    f"wait_for_session timed out after {timeout:.0f}s"
                )
            if invocation is not None:
                current = self.invocation(invocation.id)
                if current.status in {"completed", "failed", "cancelled"}:
                    if current.status == "cancelled":
                        raise SessionRuntimeError("subagent invocation was cancelled")
                    return current.output or ""
                time.sleep(0.05)
                continue
            if self._session_quiet(target_id):
                return ""
            time.sleep(0.05)

    def invocation(self, invocation_id: str) -> SubagentInvocation:
        with self._invocations_lock:
            try:
                return self._invocations[invocation_id]
            except KeyError as error:
                raise SessionRuntimeError(
                    f"unknown subagent invocation: {invocation_id}"
                ) from error

    def invocations(self) -> tuple[SubagentInvocation, ...]:
        with self._invocations_lock:
            return tuple(self._invocations.values())

    # ------------------------------------------------------------------
    # Cancellation and shutdown

    def cancel_turn(self, session_id: str, turn_id: str | None = None) -> None:
        self._ensure_workspace_session(session_id)
        info = self.store.session_info(session_id)
        target_turn = turn_id or info.active_turn_id
        if target_turn is None:
            return
        turn = self.store.turn(target_turn)
        if turn.session_id != session_id:
            raise SessionRuntimeError("turn does not belong to the requested session")
        self._cancel_turn_tree(target_turn)

    def close(self, timeout: float = 2.0) -> None:
        self._stopping = True
        with self._invocations_lock:
            for event in self._invocation_cancel.values():
                event.set()
        with self._actors_lock:
            actors = tuple(self._actors.values())
            for actor in actors:
                actor.stop = True
                if actor.cancel_event is not None:
                    actor.cancel_event.set()
                with actor.condition:
                    actor.condition.notify_all()
        deadline = time.monotonic() + max(0.0, timeout)
        for actor in actors:
            if actor.thread is not None:
                actor.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self.store.recover_session_runtime(self.workspace)
        if self._workspace_lock_file is not None:
            try:
                fcntl.flock(self._workspace_lock_file.fileno(), fcntl.LOCK_UN)
                self._workspace_lock_file.close()
            finally:
                self._workspace_lock_file = None

    @staticmethod
    def _claim_workspace(workspace: Path) -> TextIOWrapper:
        lock_path = workspace / ".litcode" / "runtime.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise SessionRuntimeError(
                "另一个 LitCode 进程正在使用当前工作区"
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        return stream

    # ------------------------------------------------------------------
    # Internal actor loop

    def _ensure_actor(self, session_id: str) -> _Actor:
        self._ensure_workspace_session(session_id)
        with self._actors_lock:
            actor = self._actors.setdefault(session_id, _Actor(session_id))
            if actor.thread is None or not actor.thread.is_alive():
                actor.stop = False
                actor.thread = threading.Thread(
                    target=self._run_actor,
                    args=(actor,),
                    name=f"session-{session_id[:8]}",
                    daemon=True,
                )
                actor.thread.start()
            return actor

    def _ensure_workspace_session(self, session_id: str) -> None:
        if not self.store.session_belongs_to(self.workspace, session_id):
            raise SessionRuntimeError(f"session is outside workspace: {session_id}")

    def _wake(self, session_id: str) -> None:
        actor = self._ensure_actor(session_id)
        with actor.condition:
            actor.condition.notify_all()

    def _run_actor(self, actor: _Actor) -> None:
        while not self._stopping and not actor.stop:
            try:
                message = self.store.claim_next_message(actor.session_id)
            except sqlite3.Error as error:
                # The UI may close its shared store while a daemon worker is
                # leaving its wait loop; there is no work left to persist.
                if self._stopping or "closed" in str(error).casefold():
                    return
                raise
            if message is None:
                with actor.condition:
                    actor.condition.wait(timeout=0.2)
                continue
            try:
                self._run_message(actor, message)
            except sqlite3.Error as error:
                if self._stopping or "closed" in str(error).casefold():
                    return
                raise

    def _run_message(self, actor: _Actor, message: QueuedMessage) -> None:
        # Legacy callers may have also projected this item into the old
        # unread inbox table.  Consuming the canonical queue advances that
        # compatibility projection without creating a second execution.
        try:
            self.store.mark_inbox_read(actor.session_id, message.id)
        except KeyError:
            pass
        invocation = self._invocation_for_message(message)
        if invocation is not None and invocation.status == "cancelled":
            # Cancellation can race between queue claim and turn creation.
            # Do not let a cancelled invocation fall through as an ordinary
            # message just because its worker claimed the row first.
            try:
                self.store.finish_message(
                    message.id, status="cancelled", result="subagent invocation cancelled"
                )
            except (KeyError, ValueError):
                pass
            return
        parent_turn = None
        invocation_cancel = None
        if invocation is not None:
            parent_turn = self.store.turn(invocation.parent_turn_id)
            invocation_cancel = self._invocation_cancel.get(invocation.id)
            self._update_invocation(invocation.id, status="running")
        root_turn_id = parent_turn.root_turn_id if parent_turn is not None else str(uuid.uuid4())
        turn = self.store.start_turn(
            actor.session_id,
            message.content,
            root_turn_id=root_turn_id,
            parent_turn_id=parent_turn.id if parent_turn is not None else None,
            status="waiting",
        )
        self.store.set_session_state(
            actor.session_id,
            status="waiting",
            activity="等待执行槽",
            active_turn_id=turn.id,
        )
        cancel_event = invocation_cancel or threading.Event()
        actor.cancel_event = cancel_event
        self._local.current_turn = turn
        self._local.cancel_event = cancel_event
        self._local.slot_turn_id = None
        try:
            if not self._acquire_slot(turn.id, cancel_event):
                self._finish_cancelled(actor, message, turn, "等待执行槽时被取消")
                return
            self._local.slot_turn_id = turn.id
            self.store.update_turn(turn.id, status="running")
            self.store.set_session_state(
                actor.session_id,
                status="running",
                activity="正在执行",
                active_turn_id=turn.id,
            )
            self.event_sink("turn_started", actor.session_id, self.store.turn(turn.id))
            session = self._agent_session(actor.session_id, actor)
            result = session.ask(message.content, cancel_event.is_set)
            self._finish_result(actor, message, turn, result)
        except Exception as error:
            self._finish_error(actor, message, turn, error)
        finally:
            actor.cancel_event = None
            self._release_current_slot(turn.id)
            self._local.current_turn = None
            self._local.cancel_event = None
            self._local.slot_turn_id = None

    def _agent_session(self, session_id: str, actor: _Actor) -> AgentSession:
        if actor.session is not None:
            return actor.session
        if self.session_factory is None:
            raise SessionRuntimeError(
                f"no AgentSession registered for {session_id}; provide session_factory"
            )
        profile = self.store.session_info(session_id).profile
        actor.session = self.session_factory(session_id, profile)
        return actor.session

    def _finish_result(
        self,
        actor: _Actor,
        message: QueuedMessage,
        turn: SessionTurn,
        result: AgentResult,
    ) -> None:
        terminal_status = "completed" if result.succeeded else "failed"
        self.store.update_turn(
            turn.id,
            status=terminal_status,
            output=result.output,
            reason=result.reason,
        )
        self.store.finish_message(
            message.id,
            status="completed" if result.succeeded else "cancelled",
            result=result.output,
        )
        self._end_actor_state(actor.session_id, terminal_status, result.reason)
        self.event_sink("turn_finished", actor.session_id, self.store.turn(turn.id))
        self._complete_invocation_for_turn(turn, result.output, result.succeeded)
        if result.reason != "completed":
            self._cancel_children(turn.id)

    def _finish_error(
        self,
        actor: _Actor,
        message: QueuedMessage,
        turn: SessionTurn,
        error: Exception,
    ) -> None:
        self.store.update_turn(
            turn.id,
            status="failed",
            output=str(error),
            reason="model_error" if error.__class__.__name__ == "ModelError" else "error",
        )
        self.store.finish_message(message.id, status="cancelled", result=str(error))
        self._end_actor_state(actor.session_id, "failed", str(error))
        self._cancel_children(turn.id)
        self._complete_invocation_for_turn(turn, str(error), False)
        self.event_sink("turn_failed", actor.session_id, self.store.turn(turn.id))

    def _finish_cancelled(
        self,
        actor: _Actor,
        message: QueuedMessage,
        turn: SessionTurn,
        reason: str,
    ) -> None:
        self.store.update_turn(turn.id, status="cancelled", reason=reason)
        self.store.finish_message(message.id, status="cancelled", result=reason)
        self._end_actor_state(actor.session_id, "cancelled", reason)
        self._cancel_children(turn.id)

    def _end_actor_state(self, session_id: str, status: str, activity: str) -> None:
        info = self.store.set_session_state(
            session_id,
            status="paused" if self.store.session_info(session_id).paused else "idle",
            activity=activity,
        )
        del info
        self._wake(session_id)

    # ------------------------------------------------------------------
    # Slot gate and invocation bookkeeping

    def _acquire_slot(self, owner: str, cancel_event: threading.Event) -> bool:
        with self._slot_condition:
            self._slot_waiters.append(owner)
            while True:
                if cancel_event.is_set() or self._stopping:
                    if owner in self._slot_waiters:
                        self._slot_waiters.remove(owner)
                    self._slot_condition.notify_all()
                    return False
                if (
                    self._slots_in_use < self.max_slots
                    and self._slot_waiters
                    and self._slot_waiters[0] == owner
                ):
                    self._slot_waiters.pop(0)
                    self._slots_in_use += 1
                    return True
                self._slot_condition.wait(timeout=0.1)

    def _release_current_slot(self, owner: str) -> bool:
        with self._slot_condition:
            try:
                slot_turn_id = self._local.slot_turn_id
            except AttributeError:
                slot_turn_id = None
            if slot_turn_id != owner:
                return False
            self._slots_in_use = max(0, self._slots_in_use - 1)
            self._local.slot_turn_id = None
            self._slot_condition.notify_all()
            return True

    def _current_turn(self) -> SessionTurn | None:
        try:
            turn = self._local.current_turn
        except AttributeError:
            turn = None
        return turn if isinstance(turn, SessionTurn) else None

    def current_cancel_event(self) -> threading.Event | None:
        try:
            value = self._local.cancel_event
        except AttributeError:
            value = None
        return value if isinstance(value, threading.Event) else None

    def request_confirmation(self, callback: Callable[[], bool]) -> bool:
        """Run a user confirmation without monopolising a model slot.

        The actor remains the single consumer of its mailbox, while the global
        execution slot is returned to the fair FIFO gate during the UI wait.
        Once the answer arrives, the actor re-enters that same gate before it
        continues its turn.
        """

        answer = self._run_with_released_slot(lambda: bool(callback()))
        return False if answer is None else answer

    def request_user_answer(self, callback: Callable[[], object]) -> object | None:
        """Wait for an arbitrary user response (question tool) with the same
        slot-release semantics as confirmation.

        Returns ``None`` when the turn was cancelled or LitCode is stopping;
        otherwise returns the callback's value.
        """

        return self._run_with_released_slot(callback)

    def _run_with_released_slot(
        self, callback: Callable[[], object]
    ) -> object | None:
        turn = self._current_turn()
        if turn is None:
            return callback()
        cancel_event = self.current_cancel_event() or threading.Event()
        released = self._release_current_slot(turn.id)
        if not released:
            return callback()
        self.store.update_turn(turn.id, status="waiting")
        self.store.set_session_state(
            turn.session_id,
            status="waiting",
            activity="等待用户回答",
            active_turn_id=turn.id,
        )
        try:
            if cancel_event.is_set() or self._stopping:
                return None
            return callback()
        finally:
            reacquired = self._acquire_slot(turn.id, cancel_event)
            if reacquired:
                self._local.slot_turn_id = turn.id
                self.store.update_turn(turn.id, status="running")
                self.store.set_session_state(
                    turn.session_id,
                    status="running",
                    activity="正在执行",
                    active_turn_id=turn.id,
                )

    def _invocation_for_message(
        self, message: QueuedMessage
    ) -> SubagentInvocation | None:
        if message.kind != "subagent_invocation" or message.source_session_id is None:
            return None
        with self._invocations_lock:
            invocations = tuple(self._invocations.values())
        for invocation in invocations:
            if (
                invocation.child_session_id == message.target_session_id
                and invocation.parent_session_id == message.source_session_id
                and invocation.prompt == message.content
            ):
                return invocation
        return None

    def _wait_for_invocation(self, invocation_id: str) -> str:
        with self._invocations_lock:
            cancel_event = self._invocation_cancel[invocation_id]
        while True:
            invocation = self.invocation(invocation_id)
            if invocation.status in {"completed", "failed", "cancelled"}:
                if invocation.status == "cancelled":
                    raise SessionRuntimeError("subagent invocation was cancelled")
                return invocation.output or ""
            if cancel_event.wait(0.05):
                raise SessionRuntimeError("subagent invocation was cancelled")

    def _update_invocation(
        self,
        invocation_id: str,
        *,
        status: InvocationStatus,
        output: str | None = None,
    ) -> None:
        with self._invocations_lock:
            previous = self._invocations[invocation_id]
            self._invocations[invocation_id] = SubagentInvocation(
                previous.id,
                previous.parent_session_id,
                previous.child_session_id,
                previous.parent_turn_id,
                previous.root_turn_id,
                previous.background,
                status,
                previous.prompt,
                previous.created_at,
                output if output is not None else previous.output,
            )

    def _complete_invocation_for_turn(
        self, turn: SessionTurn, output: str, succeeded: bool
    ) -> None:
        """Resolve the invocation which owns this child turn."""

        with self._invocations_lock:
            invocations = tuple(self._invocations.items())
        for invocation_id, invocation in invocations:
            if (
                invocation.child_session_id != turn.session_id
                or invocation.parent_turn_id != turn.parent_turn_id
                or invocation.status in {"completed", "failed", "cancelled"}
            ):
                continue
            status = "completed" if succeeded else "failed"
            self._update_invocation(invocation_id, status=status, output=output)
            if invocation.background and not self._stopping:
                child = self.store.session_info(invocation.child_session_id)
                self.store.enqueue_message(
                    invocation.parent_session_id,
                    f"子会话 {child.alias} 已完成：\n{output}",
                    source_session_id=invocation.child_session_id,
                    workspace=self.workspace,
                    kind="subagent_result",
                )
                self._wake(invocation.parent_session_id)
            return

    def _cancel_children(self, parent_turn_id: str) -> None:
        with self._invocations_lock:
            invocation_ids = tuple(self._children.get(parent_turn_id, ()))
        for invocation_id in invocation_ids:
            with self._invocations_lock:
                cancel_event = self._invocation_cancel.get(invocation_id)
                invocation = self._invocations.get(invocation_id)
                message_id = self._invocation_messages.get(invocation_id)
            if cancel_event is not None:
                cancel_event.set()
            if invocation is None:
                continue
            if invocation.status not in {"completed", "failed", "cancelled"}:
                self._update_invocation(invocation_id, status="cancelled")
            if message_id is not None:
                try:
                    self.store.cancel_queued_message(message_id)
                except (KeyError, ValueError):
                    # A worker may have claimed the item already.  Its
                    # cancellation event handles the in-flight turn.
                    pass
            self._cancel_turn_tree_for_session(invocation.child_session_id)

    def _cancel_turn_tree(self, turn_id: str) -> None:
        self._cancel_children(turn_id)
        turn = self.store.turn(turn_id)
        self._cancel_turn_tree_for_session(turn.session_id)

    def _cancel_turn_tree_for_session(self, session_id: str) -> None:
        actor = self._actors.get(session_id)
        current = self.store.session_info(session_id).active_turn_id
        if current is not None:
            # Descendants can themselves be waiting on a foreground child;
            # recurse before signalling the current worker so the complete
            # invocation tree is cancelled rather than only its root.
            self._cancel_children(current)
        if actor is not None and current is not None:
            # The worker's cancellation callback reads this event from local
            # state; this event is also retained by the actor for user stop.
            event = actor.cancel_event
            if isinstance(event, threading.Event):
                event.set()
        # A waiting/queued invocation can be cancelled before its worker has
        # constructed the local event.  Its queue item remains visible but no
        # longer gets claimed when possible.
        if current is not None:
            try:
                self.store.set_session_state(
                    session_id, activity="正在停止", status="interrupted"
                )
            except KeyError:
                pass

    def _child_system_prompt(self, profile: AgentProfile) -> str:
        suffix = profile.system_prompt.strip()
        return self.system_prompt + (f"\n\n{suffix}" if suffix else "")
