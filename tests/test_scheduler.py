from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from litcode_agent.agent import Agent
from litcode_agent.model import AssistantTurn
from litcode_agent.scheduler import Scheduler, ScheduleError, next_occurrence, normalize_schedule
from litcode_agent.session_runtime import SessionRuntime
from litcode_agent.session_store import SessionStore
from litcode_agent.tools.base import ToolExecutionContext
from litcode_agent.tools.scheduling import (
    CancelScheduledTaskTool,
    CreateScheduledTaskTool,
    ListScheduledTasksTool,
)
from litcode_agent.tools.registry import ToolRegistry


class RecordingRuntime:
    def __init__(self) -> None:
        self.notified: list[str] = []

    def notify_session(self, session_id: str) -> None:
        self.notified.append(session_id)


def epoch(value: str, zone: str = "Asia/Shanghai") -> float:
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo(zone)).timestamp()


def test_normalize_once_and_recurring_calendar_rules() -> None:
    once, zone, first = normalize_schedule(
        {
            "kind": "once",
            "timezone": "Asia/Shanghai",
            "run_at": "2027-01-02T09:30:00",
        },
        now=epoch("2027-01-01T00:00:00"),
    )
    assert once["run_at"] == "2027-01-02T09:30:00+08:00"
    assert zone == "Asia/Shanghai"
    assert first == epoch("2027-01-02T09:30:00")

    weekly, _, first = normalize_schedule(
        {
            "kind": "weekly",
            "timezone": "Asia/Shanghai",
            "time": "09:00",
            "weekdays": [5, 1, 1],
        },
        now=epoch("2027-01-04T10:00:00"),  # Monday after today's occurrence
    )
    assert weekly["weekdays"] == [1, 5]
    assert first == epoch("2027-01-08T09:00:00")


def test_monthly_rule_skips_months_without_requested_day() -> None:
    result = next_occurrence(
        {"kind": "monthly", "time": "08:00", "day_of_month": 31},
        "Asia/Shanghai",
        after=epoch("2027-04-01T00:00:00"),
    )
    assert result == epoch("2027-05-31T08:00:00")


def test_spring_dst_gap_moves_to_first_valid_minute() -> None:
    rule, _, first = normalize_schedule(
        {
            "kind": "daily",
            "timezone": "America/New_York",
            "time": "02:30",
        },
        now=epoch("2027-03-13T23:00:00", "America/New_York"),
    )
    local = datetime.fromtimestamp(first, ZoneInfo("America/New_York"))
    assert rule["time"] == "02:30"
    assert local.isoformat(timespec="minutes") == "2027-03-14T03:00-04:00"


def test_invalid_or_past_rules_are_recoverable_errors() -> None:
    with pytest.raises(ScheduleError, match="future"):
        normalize_schedule(
            {
                "kind": "once",
                "timezone": "UTC",
                "run_at": "2020-01-01T00:00:00+00:00",
            },
            now=epoch("2027-01-01T00:00:00", "UTC"),
        )
    with pytest.raises(ScheduleError, match="weekdays"):
        normalize_schedule(
            {"kind": "weekly", "timezone": "UTC", "time": "09:00"},
            now=epoch("2027-01-01T00:00:00", "UTC"),
        )


def test_due_occurrence_is_atomically_enqueued_only_once(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    creator = store.create(tmp_path, "model", [])
    target = store.create_child(creator, title="scheduled")
    due = epoch("2027-01-02T09:00:00")
    task = store.create_scheduled_task(
        creator,
        target,
        "run tests",
        {"kind": "daily", "time": "09:00"},
        "Asia/Shanghai",
        due,
    )
    runtime = RecordingRuntime()
    scheduler = Scheduler(store, runtime, tmp_path, clock=lambda: due + 5)  # type: ignore[arg-type]

    assert scheduler.dispatch_due() == (target,)
    assert scheduler.dispatch_due() == ()
    assert runtime.notified == [target]
    queue = store.queue(target)
    assert len(queue) == 1
    assert queue[0].kind == "scheduled_task"
    assert queue[0].content == "run tests"
    updated = store.scheduled_task(task.id)
    assert updated.next_run_at == epoch("2027-01-03T09:00:00")


def test_agent_tools_create_list_and_cancel_dedicated_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    creator = store.create(tmp_path, "model", [])
    context = ToolExecutionContext(creator, tmp_path.resolve())
    created = CreateScheduledTaskTool(store).execute_with_context(
        {
            "prompt": "inspect failures",
            "kind": "once",
            "timezone": "UTC",
            "run_at": "2099-01-01T00:00:00+00:00",
        },
        context,
    )
    payload = json.loads(created.content)
    assert payload["session_id"] != creator
    assert store.session_info(payload["session_id"]).parent_id == creator

    listed = ListScheduledTasksTool(store).execute_with_context({}, context)
    assert json.loads(listed.content)[0]["task_id"] == payload["task_id"]

    cancelled = CancelScheduledTaskTool(store).execute_with_context(
        {"task_id": payload["task_id"]}, context
    )
    assert json.loads(cancelled.content)["status"] == "cancelled"


def test_due_task_starts_a_complete_agent_turn(tmp_path: Path) -> None:
    class Model:
        def complete(self, messages, tools):
            assert {"role": "user", "content": "inspect repository"} in messages
            return AssistantTurn("scheduled work completed")

    store = SessionStore(tmp_path / "sessions.db")
    creator = store.create(tmp_path, "model", [])
    target = store.create_child(creator, title="scheduled")
    due = epoch("2027-01-02T09:00:00")
    store.create_scheduled_task(
        creator,
        target,
        "inspect repository",
        {"kind": "once", "run_at": "2027-01-02T09:00:00+08:00"},
        "Asia/Shanghai",
        due,
    )
    agent = Agent(
        Model(),  # type: ignore[arg-type]
        ToolRegistry([]),
        3,
        store=store,
        model_name="model",
        workspace=tmp_path,
    )
    runtime = SessionRuntime(
        store,
        tmp_path,
        session_factory=lambda session_id, profile: agent.start_session(session_id),
        claim_workspace=False,
    )
    scheduler = Scheduler(store, runtime, tmp_path, clock=lambda: due)

    assert scheduler.dispatch_due() == (target,)
    assert runtime.wait_for_idle(target, 2)
    turn = store.turns(target)[0]
    assert turn.status == "completed"
    assert turn.output == "scheduled work completed"
    runtime.close()
