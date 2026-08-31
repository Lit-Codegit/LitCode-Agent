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
from typing import Any, Sequence

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


def _synchronized(method: Any) -> Any:
    """Hold the store lock for complete execute/fetch/transaction sequences."""

    @wraps(method)
    def wrapped(self: SessionStore, *args: Any, **kwargs: Any) -> Any:
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
                messages_json TEXT NOT NULL,
                summary TEXT,
                summary_boundary INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
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
            """
        )
        self._migrate_sessions()
        with self.connection:
            self.connection.execute(
                "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', '2') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )

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
    ) -> str:
        identifier = session_id or str(uuid.uuid4())
        now = time.time()
        alias = self._new_alias(identifier, now)
        with self.connection:
            self.connection.execute(
                "INSERT INTO sessions "
                "(id, alias, workspace, title, model, parent_id, messages_json, "
                "summary, summary_boundary, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (
                    identifier,
                    alias,
                    str(workspace.resolve()),
                    title,
                    model,
                    parent_id,
                    _messages_json(messages),
                    now,
                    now,
                ),
            )
        return identifier

    @_synchronized
    def save_messages(
        self, session_id: str, messages: Sequence[Message], *, title: str | None = None
    ) -> None:
        values: list[Any] = [_messages_json(messages), time.time()]
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
            "SELECT id, alias, title, model, updated_at, parent_id FROM sessions "
            "WHERE workspace = ? ORDER BY updated_at DESC LIMIT ?",
            (str(workspace.resolve()), limit),
        ).fetchall()
        return tuple(SessionInfo(**dict(row)) for row in rows)

    @_synchronized
    def session_info(self, session_id: str) -> SessionInfo:
        row = self.connection.execute(
            "SELECT id, alias, title, model, updated_at, parent_id "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return SessionInfo(**dict(row))

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

    @_synchronized
    def update_model(self, session_id: str, model: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET model = ?, updated_at = ? WHERE id = ?",
                (model, time.time(), session_id),
            )

    def _migrate_sessions(self) -> None:
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
