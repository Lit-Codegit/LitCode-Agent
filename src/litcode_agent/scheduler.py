"""Durable calendar rules that enqueue ordinary Agent Session turns."""

from __future__ import annotations

import calendar
import os
import threading
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from litcode_agent.session_runtime import SessionRuntime
from litcode_agent.session_store import ScheduledTask, SessionStore


class ScheduleError(ValueError):
    """A visible invalid calendar rule."""


def local_timezone_name(*, platform_name: str | None = None) -> str:
    """Return an IANA name when the host exposes one, otherwise UTC."""

    configured = os.environ.get("TZ", "").lstrip(":")
    if configured:
        try:
            ZoneInfo(configured)
            return configured
        except ZoneInfoNotFoundError:
            pass
    target = os.name if platform_name is None else platform_name
    if target == "nt":
        try:
            candidate = _windows_timezone_name()
            ZoneInfo(candidate)
            return candidate
        except (ImportError, OSError, ZoneInfoNotFoundError):
            return "UTC"
    for path in (Path("/etc/localtime"), Path("/var/db/timezone/localtime")):
        try:
            resolved = str(path.resolve())
        except OSError:
            continue
        marker = "/zoneinfo/"
        if marker in resolved:
            candidate = resolved.split(marker, 1)[1]
            try:
                ZoneInfo(candidate)
                return candidate
            except ZoneInfoNotFoundError:
                pass
    return "UTC"


def _windows_timezone_name() -> str:
    """Read Windows' local zone and return its IANA mapping via tzlocal."""

    from tzlocal import get_localzone_name

    return get_localzone_name()


def normalize_schedule(
    raw: Mapping[str, object],
    *,
    now: float | None = None,
) -> tuple[dict[str, object], str, float]:
    """Validate a model-produced rule and return its first UTC occurrence."""

    kind = raw.get("kind")
    if kind not in {"once", "daily", "weekly", "monthly"}:
        raise ScheduleError("kind must be once, daily, weekly, or monthly")
    zone_name = raw.get("timezone", local_timezone_name())
    if not isinstance(zone_name, str):
        raise ScheduleError("timezone must be an IANA timezone string")
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError as error:
        raise ScheduleError(f"unknown IANA timezone: {zone_name}") from error
    current = time.time() if now is None else now

    if kind == "once":
        run_at = raw.get("run_at")
        if not isinstance(run_at, str):
            raise ScheduleError("once schedules require run_at")
        try:
            parsed = datetime.fromisoformat(run_at)
        except ValueError as error:
            raise ScheduleError("run_at must be an ISO 8601 datetime") from error
        if parsed.tzinfo is None:
            parsed = _resolve_local(parsed.date(), parsed.time(), zone)
        first = parsed.timestamp()
        if first <= current:
            raise ScheduleError("run_at must be in the future")
        return {"kind": "once", "run_at": parsed.isoformat()}, zone_name, first

    hour, minute = _parse_clock(raw.get("time"))
    normalized: dict[str, object] = {
        "kind": kind,
        "time": f"{hour:02d}:{minute:02d}",
    }
    if kind == "weekly":
        weekdays = raw.get("weekdays")
        if (
            not isinstance(weekdays, list)
            or not weekdays
            or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or not 1 <= item <= 7
                for item in weekdays
            )
        ):
            raise ScheduleError("weekly schedules require weekdays containing 1..7")
        normalized["weekdays"] = sorted(set(weekdays))
    if kind == "monthly":
        day = raw.get("day_of_month")
        if not isinstance(day, int) or isinstance(day, bool) or not 1 <= day <= 31:
            raise ScheduleError("monthly schedules require day_of_month 1..31")
        normalized["day_of_month"] = day
    first = next_occurrence(normalized, zone_name, after=current)
    if first is None:
        raise ScheduleError("schedule has no future occurrence")
    return normalized, zone_name, first


def next_occurrence(
    schedule: Mapping[str, object], timezone_name: str, *, after: float
) -> float | None:
    """Return the first occurrence strictly after an epoch timestamp."""

    kind = schedule.get("kind")
    if kind == "once":
        return None
    zone = ZoneInfo(timezone_name)
    local_after = datetime.fromtimestamp(after, zone)
    hour, minute = _parse_clock(schedule.get("time"))
    wanted_time = clock_time(hour, minute)

    if kind == "daily":
        for offset in range(0, 370):
            candidate = _resolve_local(
                local_after.date() + timedelta(days=offset), wanted_time, zone
            )
            if candidate.timestamp() > after:
                return candidate.timestamp()
    elif kind == "weekly":
        weekdays = schedule.get("weekdays")
        assert isinstance(weekdays, list)
        for offset in range(0, 15):
            candidate_date = local_after.date() + timedelta(days=offset)
            if candidate_date.isoweekday() not in weekdays:
                continue
            candidate = _resolve_local(candidate_date, wanted_time, zone)
            if candidate.timestamp() > after:
                return candidate.timestamp()
    elif kind == "monthly":
        day = schedule.get("day_of_month")
        assert isinstance(day, int)
        year, month = local_after.year, local_after.month
        for _ in range(0, 25):
            if day <= calendar.monthrange(year, month)[1]:
                candidate = _resolve_local(date(year, month, day), wanted_time, zone)
                if candidate.timestamp() > after:
                    return candidate.timestamp()
            month += 1
            if month == 13:
                year += 1
                month = 1
    else:
        raise ScheduleError(f"unknown stored schedule kind: {kind}")
    return None


def describe_task(task: ScheduledTask) -> str:
    next_run = "无"
    if task.next_run_at is not None:
        next_run = datetime.fromtimestamp(
            task.next_run_at, ZoneInfo(task.timezone)
        ).isoformat(timespec="minutes")
    return (
        f"{task.id[:8]} · {task.status} · 下次 {next_run} · "
        f"{task.prompt[:100]}"
    )


class Scheduler:
    """Small in-process dispatcher; SQLite owns all durable truth."""

    def __init__(
        self,
        store: SessionStore,
        runtime: SessionRuntime,
        workspace: Path,
        *,
        clock: Callable[[], float] = time.time,
        poll_seconds: float = 0.25,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.workspace = workspace.resolve()
        self.clock = clock
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="scheduled-agent-dispatcher", daemon=True
        )
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))

    def dispatch_due(self) -> tuple[str, ...]:
        now = self.clock()
        dispatched: list[str] = []
        for task in self.store.due_scheduled_tasks(self.workspace, now):
            assert task.next_run_at is not None
            following = next_occurrence(task.schedule, task.timezone, after=now)
            message = self.store.dispatch_scheduled_task(
                task.id, task.next_run_at, following
            )
            if message is None:
                continue
            self.runtime.notify_session(message.target_session_id)
            dispatched.append(message.target_session_id)
        return tuple(dispatched)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.dispatch_due()
            self._wake.wait(self.poll_seconds)
            self._wake.clear()


def _parse_clock(value: object) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ScheduleError("recurring schedules require time in HH:MM format")
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ScheduleError("time must use HH:MM format")
    hour, minute = (int(part) for part in parts)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ScheduleError("time must use a valid 24-hour clock")
    return hour, minute


def _resolve_local(day: date, value: clock_time, zone: ZoneInfo) -> datetime:
    """Choose fold=0; move nonexistent wall times to the next valid minute."""

    naive = datetime.combine(day, value).replace(second=0, microsecond=0)
    for offset in range(0, 181):
        candidate_naive = naive + timedelta(minutes=offset)
        candidate = candidate_naive.replace(tzinfo=zone, fold=0)
        round_trip = candidate.astimezone(timezone.utc).astimezone(zone)
        if round_trip.replace(tzinfo=None) == candidate_naive:
            return candidate
    raise ScheduleError("could not resolve local time across timezone transition")
