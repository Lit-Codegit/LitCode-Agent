"""SQLite-backed durable sessions, checkpoints, and reversible file edits."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Concatenate, Literal, Mapping, ParamSpec, Sequence, TypeVar

from litcode_agent.model import Message
from litcode_agent.tools.base import FileChange
from litcode_agent.tools.workspace import Workspace


@dataclass(frozen=True, slots=True)
class SessionInfo:
    id: str
    alias: str
    title: str
    model: str
    updated_at: float
    parent_id: str | None = None
    origin_terminal_id: str | None = None
    origin_pane_slot: int | None = None
    profile: str = "general"
    paused: bool = False
    status: str = "idle"
    activity: str = ""
    queue_size: int = 0
    active_turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class QueuedMessage:
    """A durable item in one Session's shared FIFO mailbox."""

    id: str
    sequence: int
    target_session_id: str
    source_session_id: str | None
    content: str
    kind: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: str | None = None


@dataclass(frozen=True, slots=True)
class SessionTurn:
    """Durable execution state; invocation identity lives in this table."""

    id: str
    session_id: str
    input: str
    status: str
    root_turn_id: str
    parent_turn_id: str | None
    output: str | None
    reason: str | None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


SessionScope = Literal["mounted", "current_terminal", "history"]


@dataclass(frozen=True, slots=True)
class SessionCatalogEntry:
    info: SessionInfo
    scope: SessionScope
    pane_slot: int | None = None


@dataclass(frozen=True, slots=True)
class SessionCapsule:
    alias: str
    title: str
    updated_at: float
    content: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class InboxMessage:
    id: str
    source_session_id: str
    target_session_id: str
    content: str
    created_at: float
    read: bool = False


@dataclass(frozen=True, slots=True)
class SessionReferenceSnapshot:
    id: str
    target_session_id: str
    source_session_id: str
    source_alias: str
    source_updated_at: float
    capsule: str
    created_at: float


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """A durable instruction that starts an Agent Session on a calendar."""

    id: str
    creator_session_id: str
    target_session_id: str
    prompt: str
    schedule: Mapping[str, object]
    timezone: str
    next_run_at: float | None
    status: str
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class Checkpoint:
    id: str
    label: str
    messages: tuple[Message, ...]
    file_cursor: int
    created_at: float


class _LockedConnection:
    """Serialize one SQLite connection across pane worker threads."""

    def __init__(
        self, connection: sqlite3.Connection, lock: threading.RLock
    ) -> None:
        self._connection = connection
        self._lock = lock

    def execute(
        self, sql: str, parameters: Sequence[object] = ()
    ) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.execute(sql, parameters)

    def executescript(self, sql_script: str) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.executescript(sql_script)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> _LockedConnection:
        self._lock.acquire()
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            return bool(self._connection.__exit__(exc_type, exc_value, traceback))
        finally:
            self._lock.release()


StoreParameters = ParamSpec("StoreParameters")
StoreResult = TypeVar("StoreResult")


def _synchronized(
    method: Callable[Concatenate[SessionStore, StoreParameters], StoreResult]
) -> Callable[Concatenate[SessionStore, StoreParameters], StoreResult]:
    """Hold the store lock for complete execute/fetch/transaction sequences."""

    @wraps(method)
    def wrapped(
        self: SessionStore,
        *args: StoreParameters.args,
        **kwargs: StoreParameters.kwargs,
    ) -> StoreResult:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class SessionStore:
    """A deliberately small repository; SQLite supplies atomic commits."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        self.connection = _LockedConnection(connection, self._lock)
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                alias TEXT UNIQUE,
                workspace TEXT NOT NULL,
                title TEXT NOT NULL,
                model TEXT NOT NULL,
                parent_id TEXT,
                profile TEXT NOT NULL DEFAULT 'general',
                paused INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'idle',
                activity TEXT NOT NULL DEFAULT '',
                active_turn_id TEXT,
                messages_json TEXT NOT NULL,
                summary TEXT,
                summary_boundary INTEGER,
                origin_terminal_id TEXT,
                origin_pane_slot INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_queue (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'message',
                status TEXT NOT NULL DEFAULT 'queued',
                queue_position REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                result TEXT
            );
            CREATE INDEX IF NOT EXISTS session_queue_target_idx
                ON session_queue(target_session_id, status, sequence);
            CREATE TABLE IF NOT EXISTS session_turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                input TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                root_turn_id TEXT NOT NULL,
                parent_turn_id TEXT REFERENCES session_turns(id) ON DELETE SET NULL,
                output TEXT,
                reason TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL
            );
            CREATE INDEX IF NOT EXISTS session_turns_session_idx
                ON session_turns(session_id, created_at);
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                file_cursor INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS file_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                before_content TEXT,
                after_content TEXT NOT NULL,
                before_exists INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_inbox (
                id TEXT PRIMARY KEY,
                source_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                read INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS session_references (
                id TEXT PRIMARY KEY,
                target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                source_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                source_alias TEXT NOT NULL,
                source_updated_at REAL NOT NULL,
                capsule TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                creator_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                prompt TEXT NOT NULL,
                schedule_json TEXT NOT NULL,
                timezone TEXT NOT NULL,
                next_run_at REAL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS scheduled_tasks_due_idx
                ON scheduled_tasks(status, next_run_at);
            CREATE TABLE IF NOT EXISTS scheduled_runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
                scheduled_for REAL NOT NULL,
                message_id TEXT NOT NULL UNIQUE REFERENCES session_queue(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                UNIQUE(task_id, scheduled_for)
            );
            """
        )
        self._migrate_schema()

    @_synchronized
    def close(self) -> None:
        self.connection.close()

    @_synchronized
    def create(
        self,
        workspace: Path,
        model: str,
        messages: Sequence[Message],
        *,
        session_id: str | None = None,
        title: str = "新会话",
        parent_id: str | None = None,
        profile: str = "general",
        paused: bool = False,
        origin_terminal_id: str | None = None,
        origin_pane_slot: int | None = None,
    ) -> str:
        if not profile.strip():
            raise ValueError("profile must not be empty")
        identifier = session_id or str(uuid.uuid4())
        now = time.time()
        alias = self._new_alias(identifier, now)
        with self.connection:
            self.connection.execute(
                "INSERT INTO sessions "
                "(id, alias, workspace, title, model, parent_id, profile, paused, "
                "status, activity, active_turn_id, messages_json, "
                "summary, summary_boundary, origin_terminal_id, origin_pane_slot, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', NULL, ?, NULL, NULL, ?, ?, ?, ?)",
                (
                    identifier,
                    alias,
                    str(workspace.resolve()),
                    title,
                    model,
                    parent_id,
                    profile,
                    int(paused),
                    "paused" if paused else "idle",
                    _messages_json(messages),
                    origin_terminal_id,
                    origin_pane_slot,
                    now,
                    now,
                ),
            )
        return identifier

    @_synchronized
    def save_messages(
        self, session_id: str, messages: Sequence[Message], *, title: str | None = None
    ) -> None:
        values: list[object] = [_messages_json(messages), time.time()]
        sql = "UPDATE sessions SET messages_json = ?, updated_at = ?"
        if title is not None:
            sql += ", title = ?"
            values.append(title)
        sql += " WHERE id = ?"
        values.append(session_id)
        with self.connection:
            self.connection.execute(sql, values)

    @_synchronized
    def load(self, session_id: str) -> tuple[Message, ...]:
        row = self.connection.execute(
            "SELECT messages_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return _parse_messages(row["messages_json"])

    @_synchronized
    def list_sessions(self, workspace: Path, limit: int = 50) -> tuple[SessionInfo, ...]:
        rows = self.connection.execute(
            "SELECT id, alias, title, model, updated_at, parent_id, "
            "origin_terminal_id, origin_pane_slot, profile, paused, status, activity, active_turn_id, "
            "(SELECT COUNT(*) FROM session_queue AS q WHERE q.target_session_id = sessions.id "
            "AND q.status = 'queued') AS queue_size FROM sessions "
            "WHERE workspace = ? ORDER BY updated_at DESC LIMIT ?",
            (str(workspace.resolve()), limit),
        ).fetchall()
        return tuple(_session_info(row) for row in rows)

    @_synchronized
    def session_catalog(
        self,
        workspace: Path,
        *,
        current_terminal_id: str | None,
        mounted: Mapping[str, int],
        query: str = "",
        limit: int = 50,
    ) -> tuple[SessionCatalogEntry, ...]:
        """Return one location-aware catalog shared by humans and models."""

        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        needle = query.casefold().strip()
        rows = self.connection.execute(
            "SELECT id, alias, title, model, updated_at, parent_id, "
            "origin_terminal_id, origin_pane_slot, profile, paused, status, activity, active_turn_id, "
            "(SELECT COUNT(*) FROM session_queue AS q WHERE q.target_session_id = sessions.id "
            "AND q.status = 'queued') AS queue_size FROM sessions WHERE workspace = ?",
            (str(workspace.resolve()),),
        ).fetchall()
        entries = []
        for row in rows:
            info = _session_info(row)
            if needle and needle not in info.alias.casefold() and needle not in info.title.casefold():
                continue
            pane_slot = mounted.get(info.id)
            if pane_slot is not None:
                scope: SessionScope = "mounted"
            elif current_terminal_id and info.origin_terminal_id == current_terminal_id:
                scope = "current_terminal"
            else:
                scope = "history"
            entries.append(SessionCatalogEntry(info, scope, pane_slot))
        rank = {"mounted": 0, "current_terminal": 1, "history": 2}
        entries.sort(
            key=lambda entry: (
                rank[entry.scope],
                entry.pane_slot if entry.pane_slot is not None else 99,
                -entry.info.updated_at,
                entry.info.alias,
            )
        )
        return tuple(entries[:limit])

    @_synchronized
    def session_info(self, session_id: str) -> SessionInfo:
        row = self.connection.execute(
            "SELECT id, alias, title, model, updated_at, parent_id, "
            "origin_terminal_id, origin_pane_slot, profile, paused, status, activity, active_turn_id, "
            "(SELECT COUNT(*) FROM session_queue AS q WHERE q.target_session_id = sessions.id "
            "AND q.status = 'queued') AS queue_size "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return _session_info(row)

    @_synchronized
    def session_capsule(
        self, workspace: Path, alias: str, *, max_chars: int
    ) -> SessionCapsule:
        """Build a deterministic, bounded snapshot of one local session."""

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        row = self.connection.execute(
            "SELECT alias, title, model, messages_json, summary, updated_at "
            "FROM sessions WHERE workspace = ? AND alias = ?",
            (str(workspace.resolve()), alias),
        ).fetchone()
        if row is None:
            raise KeyError(alias)
        messages = _parse_messages(row["messages_json"])
        latest_user = _latest_content(messages, "user")
        latest_assistant = _latest_content(messages, "assistant")
        parts = [f"会话 {row['alias']} · {row['title']}", f"模型：{row['model']}"]
        if row["summary"]:
            parts.append(f"已有摘要：{row['summary']}")
        if latest_user:
            parts.append(f"最近目标：{latest_user}")
        if latest_assistant:
            parts.append(f"最近结果：{latest_assistant}")
        changed = self.connection.execute(
            "SELECT DISTINCT path FROM file_changes WHERE session_id = ? "
            "ORDER BY id DESC LIMIT 12",
            (self._session_id_for_alias(workspace, alias),),
        ).fetchall()
        if changed:
            parts.append("相关文件：" + "、".join(item["path"] for item in changed))
        full = "\n".join(parts)
        return SessionCapsule(
            alias=row["alias"],
            title=row["title"],
            updated_at=row["updated_at"],
            content=full[:max_chars],
            truncated=len(full) > max_chars,
        )

    @_synchronized
    def search_session_context(
        self,
        workspace: Path,
        alias: str,
        query: str,
        *,
        max_chars: int,
    ) -> str:
        """Return matching message excerpts without disclosing full history."""

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        row = self.connection.execute(
            "SELECT messages_json FROM sessions WHERE workspace = ? AND alias = ?",
            (str(workspace.resolve()), alias),
        ).fetchone()
        if row is None:
            raise KeyError(alias)
        needle = query.casefold().strip()
        matches: list[str] = []
        for message in _parse_messages(row["messages_json"]):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if needle and needle not in content.casefold():
                continue
            role = str(message.get("role", "message"))
            matches.append(f"{role}：{content}")
        if not matches:
            return "（没有匹配的会话内容）"[:max_chars]
        return "\n\n".join(matches)[:max_chars]

    @_synchronized
    def send_to_session(
        self,
        workspace: Path,
        source_session_id: str,
        target_alias: str,
        content: str,
    ) -> InboxMessage:
        """Persist a same-workspace instruction without waking a model."""

        root = str(workspace.resolve())
        source = self.connection.execute(
            "SELECT id FROM sessions WHERE id = ? AND workspace = ?",
            (source_session_id, root),
        ).fetchone()
        target = self.connection.execute(
            "SELECT id FROM sessions WHERE alias = ? AND workspace = ?",
            (target_alias, root),
        ).fetchone()
        if source is None or target is None:
            raise KeyError(target_alias)
        message = InboxMessage(
            str(uuid.uuid4()),
            str(source["id"]),
            str(target["id"]),
            content,
            time.time(),
        )
        with self.connection:
            self.connection.execute(
                "INSERT INTO session_inbox "
                "(id, source_session_id, target_session_id, content, created_at, read) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (
                    message.id,
                    message.source_session_id,
                    message.target_session_id,
                    message.content,
                    message.created_at,
                ),
            )
            # Keep the legacy unread projection for old callers while making
            # the durable Session Queue the source consumed by new actors.
            self.connection.execute(
                "INSERT INTO session_queue "
                "(id, target_session_id, source_session_id, content, kind, status, created_at) "
                "VALUES (?, ?, ?, ?, 'message', 'queued', ?)",
                (
                    message.id,
                    message.target_session_id,
                    message.source_session_id,
                    message.content,
                    message.created_at,
                ),
            )
            self.connection.execute(
                "UPDATE session_queue SET queue_position = sequence WHERE id = ?",
                (message.id,),
            )
        return message

    @_synchronized
    def inbox(
        self, session_id: str, *, unread_only: bool = True
    ) -> tuple[InboxMessage, ...]:
        condition = " AND read = 0" if unread_only else ""
        rows = self.connection.execute(
            "SELECT id, source_session_id, target_session_id, content, "
            f"created_at, read FROM session_inbox WHERE target_session_id = ?{condition} "
            "ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return tuple(
            InboxMessage(
                row["id"],
                row["source_session_id"],
                row["target_session_id"],
                row["content"],
                row["created_at"],
                bool(row["read"]),
            )
            for row in rows
        )

    @_synchronized
    def mark_inbox_read(self, session_id: str, message_id: str) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE session_inbox SET read = 1 "
                "WHERE id = ? AND target_session_id = ?",
                (message_id, session_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(message_id)

    @_synchronized
    def record_session_reference(
        self,
        target_session_id: str,
        source_alias: str,
        source_updated_at: float,
        capsule: str,
    ) -> SessionReferenceSnapshot:
        row = self.connection.execute(
            "SELECT source.id AS source_id "
            "FROM sessions AS target JOIN sessions AS source "
            "ON source.workspace = target.workspace "
            "WHERE target.id = ? AND source.alias = ?",
            (target_session_id, source_alias),
        ).fetchone()
        if row is None:
            raise KeyError(source_alias)
        snapshot = SessionReferenceSnapshot(
            str(uuid.uuid4()),
            target_session_id,
            str(row["source_id"]),
            source_alias,
            source_updated_at,
            capsule,
            time.time(),
        )
        with self.connection:
            self.connection.execute(
                "INSERT INTO session_references VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.id,
                    snapshot.target_session_id,
                    snapshot.source_session_id,
                    snapshot.source_alias,
                    snapshot.source_updated_at,
                    snapshot.capsule,
                    snapshot.created_at,
                ),
            )
        return snapshot

    @_synchronized
    def session_references(
        self, target_session_id: str
    ) -> tuple[SessionReferenceSnapshot, ...]:
        rows = self.connection.execute(
            "SELECT * FROM session_references WHERE target_session_id = ? "
            "ORDER BY created_at",
            (target_session_id,),
        ).fetchall()
        return tuple(
            SessionReferenceSnapshot(
                row["id"],
                row["target_session_id"],
                row["source_session_id"],
                row["source_alias"],
                row["source_updated_at"],
                row["capsule"],
                row["created_at"],
            )
            for row in rows
        )

    # ------------------------------------------------------------------
    # Durable scheduled Agent tasks.  A firing and its mailbox message are
    # committed together, so a crash cannot leave one without the other.

    @_synchronized
    def create_scheduled_task(
        self,
        creator_session_id: str,
        target_session_id: str,
        prompt: str,
        schedule: Mapping[str, object],
        timezone: str,
        next_run_at: float,
    ) -> ScheduledTask:
        if not prompt.strip():
            raise ValueError("scheduled task prompt must not be empty")
        task_id = str(uuid.uuid4())
        now = time.time()
        with self.connection:
            self.connection.execute(
                "INSERT INTO scheduled_tasks "
                "(id, creator_session_id, target_session_id, prompt, schedule_json, "
                "timezone, next_run_at, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (
                    task_id,
                    creator_session_id,
                    target_session_id,
                    prompt.strip(),
                    json.dumps(schedule, ensure_ascii=False, sort_keys=True),
                    timezone,
                    next_run_at,
                    now,
                    now,
                ),
            )
        return self.scheduled_task(task_id)

    @_synchronized
    def scheduled_task(self, task_id: str) -> ScheduledTask:
        row = self.connection.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return _scheduled_task(row)

    @_synchronized
    def scheduled_tasks(
        self,
        workspace: Path,
        *,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> tuple[ScheduledTask, ...]:
        status = "" if include_inactive else "AND task.status = 'active'"
        rows = self.connection.execute(
            "SELECT task.* FROM scheduled_tasks AS task "
            "JOIN sessions AS session ON session.id = task.creator_session_id "
            f"WHERE session.workspace = ? {status} "
            "ORDER BY task.next_run_at IS NULL, task.next_run_at, task.created_at LIMIT ?",
            (str(workspace.resolve()), limit),
        ).fetchall()
        return tuple(_scheduled_task(row) for row in rows)

    @_synchronized
    def due_scheduled_tasks(
        self, workspace: Path, now: float, *, limit: int = 100
    ) -> tuple[ScheduledTask, ...]:
        rows = self.connection.execute(
            "SELECT task.* FROM scheduled_tasks AS task "
            "JOIN sessions AS session ON session.id = task.creator_session_id "
            "WHERE session.workspace = ? AND task.status = 'active' "
            "AND task.next_run_at IS NOT NULL AND task.next_run_at <= ? "
            "ORDER BY task.next_run_at, task.created_at LIMIT ?",
            (str(workspace.resolve()), now, limit),
        ).fetchall()
        return tuple(_scheduled_task(row) for row in rows)

    @_synchronized
    def dispatch_scheduled_task(
        self,
        task_id: str,
        scheduled_for: float,
        next_run_at: float | None,
    ) -> QueuedMessage | None:
        """Atomically claim one occurrence and append its Agent prompt."""

        task = self.connection.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if (
            task is None
            or task["status"] != "active"
            or task["next_run_at"] is None
            or float(task["next_run_at"]) != scheduled_for
        ):
            return None
        now = time.time()
        message_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        final_status = "completed" if next_run_at is None else "active"
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE scheduled_tasks SET next_run_at = ?, status = ?, updated_at = ? "
                "WHERE id = ? AND status = 'active' AND next_run_at = ?",
                (next_run_at, final_status, now, task_id, scheduled_for),
            )
            if cursor.rowcount != 1:
                return None
            self.connection.execute(
                "INSERT INTO session_queue "
                "(id, target_session_id, source_session_id, content, kind, status, created_at) "
                "VALUES (?, ?, ?, ?, 'scheduled_task', 'queued', ?)",
                (
                    message_id,
                    task["target_session_id"],
                    task["creator_session_id"],
                    task["prompt"],
                    now,
                ),
            )
            self.connection.execute(
                "UPDATE session_queue SET queue_position = sequence WHERE id = ?",
                (message_id,),
            )
            self.connection.execute(
                "INSERT INTO scheduled_runs "
                "(id, task_id, scheduled_for, message_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, task_id, scheduled_for, message_id, now),
            )
        return self.queue_message(message_id)

    @_synchronized
    def cancel_scheduled_task(
        self, workspace: Path, task_id: str
    ) -> ScheduledTask:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE scheduled_tasks SET status = 'cancelled', next_run_at = NULL, "
                "updated_at = ? WHERE id = ? AND status = 'active' "
                "AND creator_session_id IN (SELECT id FROM sessions WHERE workspace = ?)",
                (time.time(), task_id, str(workspace.resolve())),
            )
        if cursor.rowcount != 1:
            raise ValueError("only an active scheduled task can be cancelled")
        return self.scheduled_task(task_id)

    @_synchronized
    def update_model(self, session_id: str, model: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET model = ?, updated_at = ? WHERE id = ?",
                (model, time.time(), session_id),
            )

    @_synchronized
    def session_belongs_to(self, workspace: Path, session_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sessions WHERE id = ? AND workspace = ?",
            (session_id, str(workspace.resolve())),
        ).fetchone()
        return row is not None

    @_synchronized
    def session_id_for_reference(self, workspace: Path, reference: str) -> str:
        """Resolve a same-workspace Session by stable id or display alias."""

        root = str(workspace.resolve())
        row = self.connection.execute(
            "SELECT id FROM sessions WHERE workspace = ? AND (id = ? OR alias = ?)",
            (root, reference, reference),
        ).fetchone()
        if row is None:
            raise KeyError(reference)
        return str(row["id"])

    def _migrate_schema(self) -> None:
        row = self.connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        version = int(row["value"]) if row is not None else 0
        if version < 2:
            self._migrate_aliases()
            self._write_schema_version(2)
        if version < 3:
            self._migrate_terminal_origins()
            self._write_schema_version(3)
        if version < 4:
            # Version 4 belonged to the retired coordination repository.  Do
            # not create or mutate those legacy tables while upgrading.
            self._write_schema_version(4)
        if version < 5:
            self._write_schema_version(5)
        if version < 6:
            self._migrate_session_actors()
            self._write_schema_version(6)
        if version < 7:
            self._migrate_queue_positions()
            self._write_schema_version(7)
        if version < 8:
            self._migrate_scheduled_tasks()
            self._write_schema_version(8)

    def _migrate_aliases(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(sessions)")
        }
        if "alias" not in columns:
            with self.connection:
                self.connection.execute("ALTER TABLE sessions ADD COLUMN alias TEXT")
        rows = self.connection.execute(
            "SELECT id, created_at FROM sessions WHERE alias IS NULL OR alias = ''"
        ).fetchall()
        for row in rows:
            alias = self._new_alias(row["id"], row["created_at"])
            with self.connection:
                self.connection.execute(
                    "UPDATE sessions SET alias = ? WHERE id = ?", (alias, row["id"])
                )
        with self.connection:
            self.connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS sessions_alias_idx ON sessions(alias)"
            )

    def _migrate_terminal_origins(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(sessions)")
        }
        with self.connection:
            if "origin_terminal_id" not in columns:
                self.connection.execute(
                    "ALTER TABLE sessions ADD COLUMN origin_terminal_id TEXT"
                )
            if "origin_pane_slot" not in columns:
                self.connection.execute(
                    "ALTER TABLE sessions ADD COLUMN origin_pane_slot INTEGER"
                )

    def _write_schema_version(self, version: int) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(version),),
            )

    def _migrate_queue_positions(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(session_queue)")
        }
        if "queue_position" not in columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE session_queue ADD COLUMN "
                    "queue_position REAL NOT NULL DEFAULT 0"
                )
                self.connection.execute(
                    "UPDATE session_queue SET queue_position = sequence"
                )
        with self.connection:
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS session_queue_order_idx "
                "ON session_queue(target_session_id, status, queue_position, sequence)"
            )

    def _migrate_session_actors(self) -> None:
        """Add the mailbox actor tables without rewriting old histories."""

        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(sessions)")
        }
        additions = (
            ("profile", "TEXT NOT NULL DEFAULT 'general'"),
            ("paused", "INTEGER NOT NULL DEFAULT 0"),
            ("status", "TEXT NOT NULL DEFAULT 'idle'"),
            ("activity", "TEXT NOT NULL DEFAULT ''"),
            ("active_turn_id", "TEXT"),
        )
        with self.connection:
            for name, declaration in additions:
                if name not in columns:
                    self.connection.execute(
                        f"ALTER TABLE sessions ADD COLUMN {name} {declaration}"
                    )
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_queue (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'message',
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    result TEXT,
                    queue_position REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS session_queue_target_idx
                    ON session_queue(target_session_id, status, sequence);
                CREATE TABLE IF NOT EXISTS session_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    input TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    root_turn_id TEXT NOT NULL,
                    parent_turn_id TEXT REFERENCES session_turns(id) ON DELETE SET NULL,
                    output TEXT,
                    reason TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL
                );
                CREATE INDEX IF NOT EXISTS session_turns_session_idx
                    ON session_turns(session_id, created_at);
                """
            )

    def _migrate_scheduled_tasks(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    creator_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    prompt TEXT NOT NULL,
                    schedule_json TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    next_run_at REAL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS scheduled_tasks_due_idx
                    ON scheduled_tasks(status, next_run_at);
                CREATE TABLE IF NOT EXISTS scheduled_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
                    scheduled_for REAL NOT NULL,
                    message_id TEXT NOT NULL UNIQUE REFERENCES session_queue(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL,
                    UNIQUE(task_id, scheduled_for)
                );
                """
            )

    def _new_alias(self, identifier: str, created_at: float) -> str:
        prefix = datetime.fromtimestamp(created_at).strftime("%y%m%d-%H%M")
        attempt = 0
        while True:
            digest = hashlib.sha256(
                f"{identifier}:{attempt}".encode("utf-8")
            ).digest()
            value = int.from_bytes(digest[:2], "big") & 0x7FFF
            suffix = _crockford(value, 3)
            alias = f"{prefix}-{suffix}"
            exists = self.connection.execute(
                "SELECT 1 FROM sessions WHERE alias = ?", (alias,)
            ).fetchone()
            if exists is None:
                return alias
            attempt += 1

    def _session_id_for_alias(self, workspace: Path, alias: str) -> str:
        row = self.connection.execute(
            "SELECT id FROM sessions WHERE workspace = ? AND alias = ?",
            (str(workspace.resolve()), alias),
        ).fetchone()
        if row is None:
            raise KeyError(alias)
        return str(row["id"])

    # ------------------------------------------------------------------
    # Session Actor persistence.  This is the only mailbox API used by the
    # runtime; legacy coordination tables remain migration-only.

    @_synchronized
    def create_child(
        self,
        parent_id: str,
        model: str | None = None,
        messages: Sequence[Message] = (),
        *,
        title: str = "子会话",
        profile: str = "general",
        session_id: str | None = None,
    ) -> str:
        parent = self.connection.execute(
            "SELECT workspace, model FROM sessions WHERE id = ?", (parent_id,)
        ).fetchone()
        if parent is None:
            raise KeyError(parent_id)
        return self.create(
            Path(parent["workspace"]),
            model or str(parent["model"]),
            messages,
            session_id=session_id,
            title=title,
            parent_id=parent_id,
            profile=profile,
        )

    @_synchronized
    def children(self, session_id: str) -> tuple[SessionInfo, ...]:
        rows = self.connection.execute(
            "SELECT id, alias, title, model, updated_at, parent_id, "
            "origin_terminal_id, origin_pane_slot, profile, paused, status, activity, active_turn_id, "
            "(SELECT COUNT(*) FROM session_queue AS q WHERE q.target_session_id = sessions.id "
            "AND q.status = 'queued') AS queue_size "
            "FROM sessions WHERE parent_id = ? ORDER BY created_at, id",
            (session_id,),
        ).fetchall()
        return tuple(_session_info(row) for row in rows)

    @_synchronized
    def descendants(self, session_id: str) -> tuple[SessionInfo, ...]:
        """Return the child tree in stable depth-first creation order."""

        result: list[SessionInfo] = []

        def visit(parent_id: str) -> None:
            rows = self.connection.execute(
                "SELECT id, alias, title, model, updated_at, parent_id, "
                "origin_terminal_id, origin_pane_slot, profile, paused, status, activity, active_turn_id, "
                "(SELECT COUNT(*) FROM session_queue AS q WHERE q.target_session_id = sessions.id "
                "AND q.status = 'queued') AS queue_size "
                "FROM sessions WHERE parent_id = ? ORDER BY created_at, id",
                (parent_id,),
            ).fetchall()
            for row in rows:
                info = _session_info(row)
                result.append(info)
                visit(info.id)

        visit(session_id)
        return tuple(result)

    @_synchronized
    def session_tree(
        self, workspace: Path, *, limit: int = 500
    ) -> tuple[tuple[int, SessionInfo], ...]:
        """Return all sessions as a depth-first tree, most recently used first."""

        root = str(workspace.resolve())
        rows = self.connection.execute(
            "SELECT id, alias, title, model, updated_at, parent_id, "
            "origin_terminal_id, origin_pane_slot, profile, paused, status, activity, active_turn_id, "
            "(SELECT COUNT(*) FROM session_queue AS q WHERE q.target_session_id = sessions.id "
            "AND q.status = 'queued') AS queue_size, created_at "
            "FROM sessions WHERE workspace = ? ORDER BY updated_at DESC, id LIMIT ?",
            (root, limit),
        ).fetchall()
        infos = {str(row["id"]): _session_info(row) for row in rows}
        children: dict[str | None, list[str]] = {}
        for row in rows:
            children.setdefault(row["parent_id"], []).append(str(row["id"]))
        result: list[tuple[int, SessionInfo]] = []

        def visit(parent: str | None, depth: int) -> None:
            for identifier in children.get(parent, []):
                info = infos[identifier]
                result.append((depth, info))
                visit(identifier, depth + 1)

        visit(None, 0)
        return tuple(result)

    @_synchronized
    def session_depth(self, session_id: str) -> int:
        depth = 0
        current = session_id
        seen: set[str] = set()
        while True:
            if current in seen:
                raise RuntimeError("session parent cycle detected")
            seen.add(current)
            row = self.connection.execute(
                "SELECT parent_id FROM sessions WHERE id = ?", (current,)
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            parent = row["parent_id"]
            if parent is None:
                return depth
            depth += 1
            current = str(parent)

    @_synchronized
    def set_session_state(
        self,
        session_id: str,
        *,
        paused: bool | None = None,
        status: str | None = None,
        activity: str | None = None,
        active_turn_id: str | None = None,
    ) -> SessionInfo:
        values: list[object] = []
        assignments: list[str] = []
        if paused is not None:
            assignments.append("paused = ?")
            values.append(int(paused))
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if activity is not None:
            assignments.append("activity = ?")
            values.append(activity)
        if active_turn_id is not None:
            assignments.append("active_turn_id = ?")
            values.append(active_turn_id)
        elif status in {"idle", "interrupted", "completed", "failed", "cancelled"}:
            assignments.append("active_turn_id = NULL")
        if not assignments:
            return self.session_info(session_id)
        assignments.append("updated_at = ?")
        values.append(time.time())
        values.append(session_id)
        with self.connection:
            cursor = self.connection.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        if cursor.rowcount != 1:
            raise KeyError(session_id)
        return self.session_info(session_id)

    @_synchronized
    def set_paused(self, session_id: str, paused: bool) -> SessionInfo:
        row = self.connection.execute(
            "SELECT active_turn_id, status FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        # Pausing is a gate for the next queue item, never an interruption of
        # the turn already in flight.  Keep its running/waiting status visible
        # until the worker reaches its normal boundary.
        if paused and row["active_turn_id"] is not None:
            return self.set_session_state(
                session_id,
                paused=True,
                activity="当前轮完成后暂停",
            )
        if not paused and row["active_turn_id"] is not None:
            return self.set_session_state(
                session_id,
                paused=False,
                activity="正在执行",
            )
        return self.set_session_state(
            session_id,
            paused=paused,
            status="paused" if paused else "idle",
            activity="已暂停" if paused else "等待队列",
        )

    @_synchronized
    def enqueue_message(
        self,
        target_session_id: str,
        content: str,
        *,
        source_session_id: str | None = None,
        workspace: Path | None = None,
        kind: str = "message",
        message_id: str | None = None,
    ) -> QueuedMessage:
        content = content.strip()
        if not content:
            raise ValueError("queued message content must not be empty")
        target = self.connection.execute(
            "SELECT workspace FROM sessions WHERE id = ?", (target_session_id,)
        ).fetchone()
        if target is None:
            raise KeyError(target_session_id)
        root = str(workspace.resolve()) if workspace is not None else None
        if root is not None and target["workspace"] != root:
            raise KeyError(target_session_id)
        if source_session_id is not None:
            source = self.connection.execute(
                "SELECT workspace FROM sessions WHERE id = ?", (source_session_id,)
            ).fetchone()
            if source is None or source["workspace"] != target["workspace"]:
                raise KeyError(source_session_id)
        identifier = message_id or str(uuid.uuid4())
        created_at = time.time()
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO session_queue "
                "(id, target_session_id, source_session_id, content, kind, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?)",
                (
                    identifier,
                    target_session_id,
                    source_session_id,
                    content,
                    kind,
                    created_at,
                ),
            )
            self.connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (created_at, target_session_id),
            )
            self.connection.execute(
                "UPDATE session_queue SET queue_position = sequence WHERE id = ?",
                (identifier,),
            )
        row = self.connection.execute(
            "SELECT * FROM session_queue WHERE sequence = ?", (cursor.lastrowid,)
        ).fetchone()
        assert row is not None
        return _queued_message(row)

    @_synchronized
    def queue(
        self,
        session_id: str,
        *,
        include_finished: bool = False,
        limit: int = 100,
    ) -> tuple[QueuedMessage, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        statuses = "('queued', 'running', 'completed', 'cancelled')" if include_finished else "('queued', 'running')"
        rows = self.connection.execute(
            "SELECT * FROM session_queue WHERE target_session_id = ? "
            f"AND status IN {statuses} ORDER BY queue_position, sequence LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return tuple(_queued_message(row) for row in rows)

    @_synchronized
    def queue_message(self, message_id: str) -> QueuedMessage:
        row = self.connection.execute(
            "SELECT * FROM session_queue WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise KeyError(message_id)
        return _queued_message(row)

    @_synchronized
    def claim_next_message(self, session_id: str) -> QueuedMessage | None:
        """Atomically claim the queue head; only a Session runner may call it."""

        session = self.connection.execute(
            "SELECT paused, active_turn_id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise KeyError(session_id)
        if session["paused"] or session["active_turn_id"] is not None:
            return None
        row = self.connection.execute(
            "SELECT * FROM session_queue WHERE target_session_id = ? "
            "AND status = 'queued' ORDER BY queue_position, sequence LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        started_at = time.time()
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE session_queue SET status = 'running', started_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (started_at, row["id"]),
            )
        if cursor.rowcount != 1:
            return None
        claimed = self.connection.execute(
            "SELECT * FROM session_queue WHERE id = ?", (row["id"],)
        ).fetchone()
        assert claimed is not None
        return _queued_message(claimed)

    @_synchronized
    def finish_message(
        self,
        message_id: str,
        *,
        status: str = "completed",
        result: str | None = None,
    ) -> QueuedMessage:
        if status not in {"completed", "cancelled"}:
            raise ValueError("finished message status must be completed or cancelled")
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE session_queue SET status = ?, finished_at = ?, result = ? "
                "WHERE id = ? AND status = 'running'",
                (status, time.time(), result, message_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(message_id)
        return self.queue_message(message_id)

    @_synchronized
    def cancel_queued_message(self, message_id: str) -> QueuedMessage:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE session_queue SET status = 'cancelled', finished_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (time.time(), message_id),
            )
        if cursor.rowcount != 1:
            raise ValueError("only a queued message can be cancelled")
        return self.queue_message(message_id)

    @_synchronized
    def reorder_queued_message(
        self, session_id: str, message_id: str, before_message_id: str
    ) -> QueuedMessage:
        """Move one queued item before another item in the same mailbox."""

        rows = self.connection.execute(
            "SELECT id, queue_position, sequence FROM session_queue "
            "WHERE target_session_id = ? AND status = 'queued' "
            "ORDER BY queue_position, sequence",
            (session_id,),
        ).fetchall()
        ordered = [str(row["id"]) for row in rows]
        if message_id not in ordered or before_message_id not in ordered:
            raise ValueError("both messages must be queued in the same session")
        if message_id == before_message_id:
            raise ValueError("a message cannot be moved before itself")
        remaining = [identifier for identifier in ordered if identifier != message_id]
        before_index = remaining.index(before_message_id)
        previous_id = remaining[before_index - 1] if before_index else None
        next_id = before_message_id
        positions = {str(row["id"]): float(row["queue_position"]) for row in rows}
        previous = positions[previous_id] if previous_id is not None else None
        following = positions[next_id]
        position = following - 1.0 if previous is None else (previous + following) / 2.0
        if (previous is not None and position == previous) or position == following:
            self._renumber_queue(session_id, remaining)
            rows = self.connection.execute(
                "SELECT id, queue_position FROM session_queue "
                "WHERE target_session_id = ? AND status = 'queued' "
                "ORDER BY queue_position, sequence",
                (session_id,),
            ).fetchall()
            positions = {str(row["id"]): float(row["queue_position"]) for row in rows}
            previous = positions[previous_id] if previous_id is not None else None
            following = positions[next_id]
            position = following - 1.0 if previous is None else (previous + following) / 2.0
        with self.connection:
            self.connection.execute(
                "UPDATE session_queue SET queue_position = ? "
                "WHERE id = ? AND target_session_id = ? AND status = 'queued'",
                (position, message_id, session_id),
            )
        return self.queue_message(message_id)

    @_synchronized
    def move_queued_message(
        self, session_id: str, message_id: str, direction: int
    ) -> QueuedMessage:
        """Move a queued item one place up (-1) or down (+1)."""

        if direction not in {-1, 1}:
            raise ValueError("queue direction must be -1 or 1")
        rows = self.connection.execute(
            "SELECT id FROM session_queue WHERE target_session_id = ? "
            "AND status = 'queued' ORDER BY queue_position, sequence",
            (session_id,),
        ).fetchall()
        ordered = [str(row["id"]) for row in rows]
        if message_id not in ordered:
            raise ValueError("only a queued message can be moved")
        index = ordered.index(message_id)
        target_index = index + direction
        if not 0 <= target_index < len(ordered):
            return self.queue_message(message_id)
        before_id = ordered[target_index] if direction < 0 else (
            ordered[target_index + 1] if target_index + 1 < len(ordered) else None
        )
        if before_id is None:
            # Put the item after the current queue tail by moving the last item
            # before it and preserving the same public operation semantics.
            last_id = ordered[-1]
            positions = self.connection.execute(
                "SELECT queue_position FROM session_queue WHERE id = ?", (last_id,)
            ).fetchone()
            assert positions is not None
            with self.connection:
                self.connection.execute(
                    "UPDATE session_queue SET queue_position = ? WHERE id = ?",
                    (float(positions["queue_position"]) + 1.0, message_id),
                )
            return self.queue_message(message_id)
        return self.reorder_queued_message(session_id, message_id, before_id)

    def _renumber_queue(self, session_id: str, ordered: list[str]) -> None:
        """Restore roomy positions after many midpoint moves."""

        with self.connection:
            for index, identifier in enumerate(ordered, start=1):
                self.connection.execute(
                    "UPDATE session_queue SET queue_position = ? "
                    "WHERE target_session_id = ? AND id = ? AND status = 'queued'",
                    (float(index), session_id, identifier),
                )

    @_synchronized
    def start_turn(
        self,
        session_id: str,
        input_text: str,
        *,
        parent_turn_id: str | None = None,
        root_turn_id: str | None = None,
        turn_id: str | None = None,
        status: str = "queued",
    ) -> SessionTurn:
        identifier = turn_id or str(uuid.uuid4())
        root = root_turn_id or identifier
        created_at = time.time()
        with self.connection:
            self.connection.execute(
                "INSERT INTO session_turns "
                "(id, session_id, input, status, root_turn_id, parent_turn_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    session_id,
                    input_text,
                    status,
                    root,
                    parent_turn_id,
                    created_at,
                ),
            )
        return self.turn(identifier)

    @_synchronized
    def turn(self, turn_id: str) -> SessionTurn:
        row = self.connection.execute(
            "SELECT * FROM session_turns WHERE id = ?", (turn_id,)
        ).fetchone()
        if row is None:
            raise KeyError(turn_id)
        return _session_turn(row)

    @_synchronized
    def turns(
        self, session_id: str, *, limit: int = 100
    ) -> tuple[SessionTurn, ...]:
        rows = self.connection.execute(
            "SELECT * FROM session_turns WHERE session_id = ? "
            "ORDER BY created_at, id LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return tuple(_session_turn(row) for row in rows)

    @_synchronized
    def update_turn(
        self,
        turn_id: str,
        *,
        status: str | None = None,
        output: str | None = None,
        reason: str | None = None,
    ) -> SessionTurn:
        assignments: list[str] = []
        values: list[object] = []
        if status is not None:
            assignments.extend(["status = ?", "started_at = COALESCE(started_at, ?)"])
            values.extend([status, time.time()])
            if status in {"completed", "failed", "cancelled", "interrupted"}:
                assignments.extend(["finished_at = ?"])
                values.append(time.time())
        if output is not None:
            assignments.append("output = ?")
            values.append(output)
        if reason is not None:
            assignments.append("reason = ?")
            values.append(reason)
        if not assignments:
            return self.turn(turn_id)
        values.append(turn_id)
        with self.connection:
            cursor = self.connection.execute(
                f"UPDATE session_turns SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        if cursor.rowcount != 1:
            raise KeyError(turn_id)
        return self.turn(turn_id)

    @_synchronized
    def recover_session_runtime(self, workspace: Path) -> tuple[SessionTurn, ...]:
        """Mark active turns interrupted while retaining all messages/queue items."""

        rows = self.connection.execute(
            "SELECT turn.* FROM session_turns AS turn "
            "JOIN sessions AS session ON session.id = turn.session_id "
            "WHERE session.workspace = ? AND turn.status IN ('queued', 'running', 'waiting')",
            (str(workspace.resolve()),),
        ).fetchall()
        recovered: list[SessionTurn] = []
        now = time.time()
        with self.connection:
            for row in rows:
                self.connection.execute(
                    "UPDATE session_turns SET status = 'interrupted', reason = ?, "
                    "finished_at = ? WHERE id = ?",
                    ("LitCode 进程退出", now, row["id"]),
                )
            self.connection.execute(
                "UPDATE sessions SET status = 'interrupted', active_turn_id = NULL, "
                "activity = '上次进程退出时中断' , updated_at = ? "
                "WHERE workspace = ? AND status IN ('queued', 'waiting', 'running')",
                (now, str(workspace.resolve())),
            )
            self.connection.execute(
                "UPDATE session_queue SET status = 'queued', started_at = NULL "
                "WHERE target_session_id IN (SELECT id FROM sessions WHERE workspace = ?) "
                "AND status = 'running'",
                (str(workspace.resolve()),),
            )
        for row in rows:
            recovered.append(self.turn(row["id"]))
        return tuple(recovered)

    @_synchronized
    def add_checkpoint(
        self, session_id: str, label: str, messages: Sequence[Message]
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            str(uuid.uuid4()),
            label,
            tuple(messages),
            self.file_cursor(session_id),
            time.time(),
        )
        with self.connection:
            self.connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                (
                    checkpoint.id,
                    session_id,
                    label,
                    _messages_json(messages),
                    checkpoint.file_cursor,
                    checkpoint.created_at,
                ),
            )
        return checkpoint

    @_synchronized
    def checkpoints(self, session_id: str) -> tuple[Checkpoint, ...]:
        rows = self.connection.execute(
            "SELECT id, label, messages_json, file_cursor, created_at "
            "FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return tuple(
            Checkpoint(
                row["id"],
                row["label"],
                _parse_messages(row["messages_json"]),
                row["file_cursor"],
                row["created_at"],
            )
            for row in rows
        )

    @_synchronized
    def discard_checkpoints_after(self, session_id: str, created_at: float) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM checkpoints WHERE session_id = ? AND created_at > ?",
                (session_id, created_at),
            )

    @_synchronized
    def save_summary(self, session_id: str, summary: str, boundary: int) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET summary = ?, summary_boundary = ?, updated_at = ? "
                "WHERE id = ?",
                (summary, boundary, time.time(), session_id),
            )

    @_synchronized
    def clear_summary(self, session_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET summary = NULL, summary_boundary = NULL, "
                "updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )

    @_synchronized
    def summary(self, session_id: str) -> tuple[str, int] | None:
        row = self.connection.execute(
            "SELECT summary, summary_boundary FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None or row["summary"] is None:
            return None
        return row["summary"], row["summary_boundary"]

    @_synchronized
    def record_change(self, session_id: str, change: FileChange) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO file_changes(session_id, path, before_content, "
                "after_content, before_exists) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    change.path,
                    change.before_content,
                    change.after_content,
                    int(change.before_exists),
                ),
            )

    @_synchronized
    def discard_changes_after(self, session_id: str, cursor: int) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM file_changes WHERE session_id = ? AND id > ?",
                (session_id, cursor),
            )

    @_synchronized
    def file_cursor(self, session_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(id), 0) AS cursor FROM file_changes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["cursor"])

    @_synchronized
    def restore_files(
        self, session_id: str, cursor: int, workspace: Workspace, *, forward: bool = False
    ) -> int:
        operator = ">"
        order = "ASC" if forward else "DESC"
        rows = self.connection.execute(
            f"SELECT * FROM file_changes WHERE session_id = ? AND id {operator} ? "
            f"ORDER BY id {order}",
            (session_id, cursor),
        ).fetchall()
        changes = [dict(row) for row in rows]
        virtual: dict[str, str | None] = {}
        for change in changes:
            path = workspace.resolve(change["path"], must_exist=False)
            actual = virtual.get(change["path"], _read_optional(path))
            expected = (
                change["before_content"] if forward else change["after_content"]
            )
            if actual != expected:
                raise RuntimeError(
                    f"文件已在 Agent 编辑后发生变化，拒绝覆盖：{change['path']}"
                )
            virtual[change["path"]] = (
                change["after_content"]
                if forward
                else (change["before_content"] if change["before_exists"] else None)
            )
        for raw_path, content in virtual.items():
            path = workspace.resolve(raw_path, must_exist=False)
            if content is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write(path, content)
        return len(virtual)

    @_synchronized
    def fork(self, source_id: str, checkpoint: Checkpoint, model: str) -> str:
        source = self.connection.execute(
            "SELECT workspace FROM sessions WHERE id = ?", (source_id,)
        ).fetchone()
        if source is None:
            raise KeyError(source_id)
        identifier = self.create(
            Path(source["workspace"]),
            model,
            checkpoint.messages,
            title=f"分支 · {checkpoint.label}",
            parent_id=source_id,
        )
        self.add_checkpoint(identifier, f"分支起点 · {checkpoint.label}", checkpoint.messages)
        return identifier


def _messages_json(messages: Sequence[Message]) -> str:
    return json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))


def _parse_messages(raw: str) -> tuple[Message, ...]:
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("invalid stored messages")
    return tuple(value)


def _latest_content(messages: Sequence[Message], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _session_info(row: sqlite3.Row) -> SessionInfo:
    return SessionInfo(
        id=str(row["id"]),
        alias=str(row["alias"]),
        title=str(row["title"]),
        model=str(row["model"]),
        updated_at=float(row["updated_at"]),
        parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
        origin_terminal_id=(
            str(row["origin_terminal_id"])
            if row["origin_terminal_id"] is not None
            else None
        ),
        origin_pane_slot=(
            int(row["origin_pane_slot"])
            if row["origin_pane_slot"] is not None
            else None
        ),
        profile=str(row["profile"]) if row["profile"] is not None else "general",
        paused=bool(row["paused"]),
        status=str(row["status"]) if row["status"] is not None else "idle",
        activity=str(row["activity"]) if row["activity"] is not None else "",
        queue_size=int(row["queue_size"]) if row["queue_size"] is not None else 0,
        active_turn_id=(
            str(row["active_turn_id"]) if row["active_turn_id"] is not None else None
        ),
    )


def _queued_message(row: sqlite3.Row) -> QueuedMessage:
    return QueuedMessage(
        id=str(row["id"]),
        sequence=int(row["sequence"]),
        target_session_id=str(row["target_session_id"]),
        source_session_id=(
            str(row["source_session_id"])
            if row["source_session_id"] is not None
            else None
        ),
        content=str(row["content"]),
        kind=str(row["kind"]),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        started_at=(
            float(row["started_at"]) if row["started_at"] is not None else None
        ),
        finished_at=(
            float(row["finished_at"]) if row["finished_at"] is not None else None
        ),
        result=str(row["result"]) if row["result"] is not None else None,
    )


def _scheduled_task(row: sqlite3.Row) -> ScheduledTask:
    schedule = json.loads(row["schedule_json"])
    if not isinstance(schedule, dict):
        raise ValueError("stored schedule must be an object")
    return ScheduledTask(
        id=str(row["id"]),
        creator_session_id=str(row["creator_session_id"]),
        target_session_id=str(row["target_session_id"]),
        prompt=str(row["prompt"]),
        schedule=schedule,
        timezone=str(row["timezone"]),
        next_run_at=(
            float(row["next_run_at"]) if row["next_run_at"] is not None else None
        ),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _session_turn(row: sqlite3.Row) -> SessionTurn:
    return SessionTurn(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        input=str(row["input"]),
        status=str(row["status"]),
        root_turn_id=str(row["root_turn_id"]),
        parent_turn_id=(
            str(row["parent_turn_id"]) if row["parent_turn_id"] is not None else None
        ),
        output=str(row["output"]) if row["output"] is not None else None,
        reason=str(row["reason"]) if row["reason"] is not None else None,
        created_at=float(row["created_at"]),
        started_at=(
            float(row["started_at"]) if row["started_at"] is not None else None
        ),
        finished_at=(
            float(row["finished_at"]) if row["finished_at"] is not None else None
        ),
    )


def _crockford(value: int, width: int) -> str:
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    chars: list[str] = []
    for _ in range(width):
        value, remainder = divmod(value, len(alphabet))
        chars.append(alphabet[remainder])
    return "".join(reversed(chars))


def _read_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(content)
            temporary_name = temporary.name
        if mode is not None:
            os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
