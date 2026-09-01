from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from litcode_agent.agent import AgentResult
from litcode_agent.session_runtime import SessionRuntime, SessionRuntimeError
from litcode_agent.session_store import SessionStore


class FakeSession:
    def __init__(self, session_id: str, runtime: SessionRuntime | None = None) -> None:
        self.session_id = session_id
        self.runtime = runtime
        self.calls: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.spawn_foreground = False

    def ask(self, prompt: str, should_cancel) -> AgentResult:
        self.calls.append(prompt)
        self.started.set()
        if self.spawn_foreground:
            self.spawn_foreground = False
            assert self.runtime is not None
            child = self.runtime.spawn_subagent(
                self.session_id, "child work", background=False
            )
            return AgentResult(f"child said: {child.output}", "completed", 1, ())
        while not self.release.wait(0.01):
            if should_cancel():
                return AgentResult("cancelled", "cancelled", 1, ())
        return AgentResult(prompt.upper(), "completed", 1, ())

    def close(self, *args: object) -> None:
        return None


def test_session_runtime_consumes_one_fifo_queue(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create(tmp_path, "model", [])
    made: dict[str, FakeSession] = {}

    def factory(identifier: str, profile: str) -> FakeSession:
        del profile
        made[identifier] = FakeSession(identifier)
        made[identifier].release.set()
        return made[identifier]

    runtime = SessionRuntime(store, tmp_path, factory)
    runtime.submit(session_id, "one")
    runtime.submit(session_id, "two")

    assert runtime.wait_for_idle(session_id)
    assert made[session_id].calls == ["one", "two"]
    assert [turn.input for turn in store.turns(session_id)] == ["one", "two"]
    assert all(item.status == "completed" for item in store.queue(session_id, include_finished=True))
    runtime.close()


def test_pause_keeps_queue_and_resume_wakes_it(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create(tmp_path, "model", [])
    made = FakeSession(session_id)
    made.release.set()
    runtime = SessionRuntime(store, tmp_path, lambda *_: made)
    runtime.pause(session_id)
    runtime.submit(session_id, "wait")
    time.sleep(0.05)
    assert made.calls == []
    assert store.session_info(session_id).paused
    runtime.resume(session_id)
    assert runtime.wait_for_idle(session_id)
    assert made.calls == ["wait"]
    runtime.close()


def test_pause_does_not_interrupt_the_active_turn(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create(tmp_path, "model", [])
    made = FakeSession(session_id)
    runtime = SessionRuntime(store, tmp_path, lambda *_: made)
    runtime.submit(session_id, "long")
    assert made.started.wait(1)

    runtime.pause(session_id)
    assert store.session_info(session_id).paused
    assert store.session_info(session_id).active_turn_id is not None
    assert not runtime.wait_for_idle(session_id, timeout=0.05)

    made.release.set()
    assert runtime.wait_for_idle(session_id)
    assert store.session_info(session_id).status == "paused"
    runtime.close()


def test_wait_for_session_blocks_until_background_invocation_finishes(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    parent_id = store.create(tmp_path, "model", [])
    child_started = threading.Event()
    child_release = threading.Event()
    made: dict[str, FakeSession] = {}
    runtime: SessionRuntime

    class Parent(FakeSession):
        def __init__(self, session_id: str, runtime: SessionRuntime) -> None:
            super().__init__(session_id, runtime)
            self.spawned = False

        def ask(self, prompt: str, should_cancel) -> AgentResult:
            if self.spawned:
                return AgentResult("result note", "completed", 1, ())
            self.spawned = True
            result = runtime.spawn_subagent(
                self.session_id, "background", background=True
            )
            output = runtime.wait_for_session(result.invocation_id, timeout=10)
            return AgentResult(f"waited:{output}", "completed", 2, ())

    class Child(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            child_started.set()
            if not child_release.wait(1):
                return AgentResult("timed out", "cancelled", 1, ())
            return AgentResult("child done", "completed", 1, ())

    def factory(identifier: str, profile: str) -> FakeSession:
        del profile
        session: FakeSession = (
            Parent(identifier, runtime)
            if identifier == parent_id
            else Child(identifier, runtime)
        )
        made[identifier] = session
        return session

    runtime = SessionRuntime(store, tmp_path, factory, max_slots=1)
    runtime.submit(parent_id, "parent")
    assert child_started.wait(5)
    child_release.set()
    assert runtime.wait_for_idle(parent_id, 5)
    assert any(
        "waited:child done" in (turn.output or "")
        for turn in store.turns(parent_id)
    )
    runtime.close()


def test_wait_for_session_rejects_self_unknown_and_times_out(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    parent_id = store.create(tmp_path, "model", [])
    busy_id = store.create(tmp_path, "model", [])
    answers: list[str] = []
    runtime: SessionRuntime

    class ErrorProbe(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            self.calls.append(prompt)
            try:
                runtime.wait_for_session(prompt, 0.2)
            except SessionRuntimeError as error:
                answers.append(str(error))
            return AgentResult("done", "completed", 1, ())

    made = ErrorProbe(parent_id)
    runtime = SessionRuntime(store, tmp_path, lambda *_: made)
    store.set_session_state(busy_id, status="running", activity="卡住")
    runtime.submit(parent_id, parent_id)
    runtime.submit(parent_id, "no-such-session")
    runtime.submit(parent_id, busy_id)
    assert runtime.wait_for_idle(parent_id, 5)
    assert "cannot wait for itself" in answers[0]
    assert "找不到会话" in answers[1]
    assert "timed out" in answers[2]
    runtime.close()


def test_foreground_child_releases_slot_and_background_result_queues(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    parent_id = store.create(tmp_path, "model", [])
    made: dict[str, FakeSession] = {}
    runtime: SessionRuntime

    def factory(identifier: str, profile: str) -> FakeSession:
        del profile
        session = FakeSession(identifier, runtime)
        session.release.set()
        made[identifier] = session
        return session

    runtime = SessionRuntime(store, tmp_path, factory, max_slots=1)
    made_parent = FakeSession(parent_id, runtime)
    made_parent.release.set()
    made_parent.spawn_foreground = True
    made[parent_id] = made_parent
    runtime.register(parent_id, made_parent)
    runtime.submit(parent_id, "parent")

    assert runtime.wait_for_idle(parent_id)
    assert made_parent.calls == ["parent"]
    assert "child said: CHILD WORK" in (store.turns(parent_id)[-1].output or "")

    runtime.close()


def test_depth_and_budget_are_visible_errors(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    root = store.create(tmp_path, "model", [])
    runtime = SessionRuntime(store, tmp_path, lambda *_: None, max_depth=0, invocation_budget=1)
    # No active Agent turn means direct calls are rejected rather than making
    # an unowned child; this protects the invocation ownership boundary.
    try:
        runtime.spawn_subagent(root, "no parent")
    except SessionRuntimeError as error:
        assert "active parent turn" in str(error)
    else:
        raise AssertionError("spawn without an active turn should fail")
    runtime.close()


def test_cancel_parent_cascades_to_a_running_background_child(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    parent_id = store.create(tmp_path, "model", [])
    child_started = threading.Event()
    parent_started = threading.Event()
    made: dict[str, FakeSession] = {}
    runtime: SessionRuntime

    class Parent(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            parent_started.set()
            result = runtime.spawn_subagent(
                self.session_id, "background", background=True
            )
            assert result.background
            while not should_cancel():
                time.sleep(0.01)
            return AgentResult("cancelled", "cancelled", 1, ())

    class Child(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            child_started.set()
            while not should_cancel():
                time.sleep(0.01)
            return AgentResult("cancelled", "cancelled", 1, ())

    def factory(identifier: str, profile: str) -> FakeSession:
        del profile
        session = Parent(identifier, runtime) if identifier == parent_id else Child(identifier, runtime)
        made[identifier] = session
        return session

    runtime = SessionRuntime(store, tmp_path, factory, max_slots=2)
    runtime.submit(parent_id, "parent")
    assert parent_started.wait(1)
    assert child_started.wait(1)
    runtime.cancel_turn(parent_id)
    assert runtime.wait_for_idle(parent_id)
    child_id = next(identifier for identifier in made if identifier != parent_id)
    assert runtime.wait_for_idle(child_id)
    assert any(item.status == "cancelled" for item in runtime.invocations())
    assert store.session_info(child_id).parent_id == parent_id
    runtime.close()


def test_confirmation_releases_slot_and_reacquires_fifo(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    first_id = store.create(tmp_path, "model", [])
    second_id = store.create(tmp_path, "model", [])
    confirmation_entered = threading.Event()
    allow_confirmation = threading.Event()
    second_started = threading.Event()
    made: dict[str, FakeSession] = {}
    runtime: SessionRuntime

    class ConfirmingSession(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            self.calls.append(prompt)
            confirmation_entered.set()
            approved = runtime.request_confirmation(
                lambda: allow_confirmation.wait(timeout=1)
            )
            return AgentResult("approved" if approved else "denied", "completed", 1, ())

    class QuickSession(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            self.calls.append(prompt)
            second_started.set()
            return AgentResult("second", "completed", 1, ())

    def factory(identifier: str, profile: str) -> FakeSession:
        del profile
        session = (
            ConfirmingSession(identifier, runtime)
            if identifier == first_id
            else QuickSession(identifier, runtime)
        )
        made[identifier] = session
        return session

    runtime = SessionRuntime(store, tmp_path, factory, max_slots=1)
    runtime.submit(first_id, "confirm")
    assert confirmation_entered.wait(1)
    runtime.submit(second_id, "quick")
    assert second_started.wait(1)
    assert not runtime.wait_for_idle(first_id, timeout=0.05)
    allow_confirmation.set()
    assert runtime.wait_for_idle(first_id)
    assert runtime.wait_for_idle(second_id)
    assert made[first_id].calls == ["confirm"]


def test_request_user_answer_releases_slot_and_returns_value(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    first_id = store.create(tmp_path, "model", [])
    second_id = store.create(tmp_path, "model", [])
    question_entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    made: dict[str, FakeSession] = {}
    runtime: SessionRuntime

    class QuestionSession(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            self.calls.append(prompt)

            def wait_answer() -> object:
                question_entered.set()
                return release.wait(timeout=3)

            value = runtime.request_user_answer(wait_answer)
            return AgentResult(f"answer:{value}", "completed", 1, ())

    class QuickSession(FakeSession):
        def ask(self, prompt: str, should_cancel) -> AgentResult:
            self.calls.append(prompt)
            second_started.set()
            return AgentResult("second", "completed", 1, ())

    def factory(identifier: str, profile: str) -> FakeSession:
        del profile
        session = (
            QuestionSession(identifier, runtime)
            if identifier == first_id
            else QuickSession(identifier, runtime)
        )
        made[identifier] = session
        return session

    runtime = SessionRuntime(store, tmp_path, factory, max_slots=1)
    runtime.submit(first_id, "ask")
    assert question_entered.wait(1)
    runtime.submit(second_id, "quick")
    assert second_started.wait(1)
    assert not runtime.wait_for_idle(first_id, timeout=0.05)
    release.set()
    assert runtime.wait_for_idle(first_id)
    assert runtime.wait_for_idle(second_id)
    assert made[first_id].calls == ["ask"]
    assert store.turns(first_id)[0].output == "answer:True"
    assert made[second_id].calls == ["quick"]
    runtime.close()
