"""SQLite-backed durable sessions, checkpoints, and reversible file edits."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from litcode_agent.model import Message
from litcode_agent.tools.base import FileChange
from litcode_agent.tools.workspace import Workspace


@dataclass(frozen=True, slots=True)
class SessionInfo:
    id: str
    title: str
    model: str
    updated_at: float
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class Checkpoint:
    id: str
    label: str
    messages: tuple[Message, ...]
    file_cursor: int
    created_at: float


class SessionStore:
    """A deliberately small repository; SQLite supplies atomic commits."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
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
            """
        )

    def close(self) -> None:
        self.connection.close()

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
        with self.connection:
            self.connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (
                    identifier,
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

    def load(self, session_id: str) -> tuple[Message, ...]:
        row = self.connection.execute(
            "SELECT messages_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return _parse_messages(row["messages_json"])

    def list_sessions(self, workspace: Path, limit: int = 50) -> tuple[SessionInfo, ...]:
        rows = self.connection.execute(
            "SELECT id, title, model, updated_at, parent_id FROM sessions "
            "WHERE workspace = ? ORDER BY updated_at DESC LIMIT ?",
            (str(workspace.resolve()), limit),
        ).fetchall()
        return tuple(SessionInfo(**dict(row)) for row in rows)

    def session_info(self, session_id: str) -> SessionInfo:
        row = self.connection.execute(
            "SELECT id, title, model, updated_at, parent_id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return SessionInfo(**dict(row))

    def update_model(self, session_id: str, model: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET model = ?, updated_at = ? WHERE id = ?",
                (model, time.time(), session_id),
            )

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

    def discard_checkpoints_after(self, session_id: str, created_at: float) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM checkpoints WHERE session_id = ? AND created_at > ?",
                (session_id, created_at),
            )

    def save_summary(self, session_id: str, summary: str, boundary: int) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET summary = ?, summary_boundary = ?, updated_at = ? "
                "WHERE id = ?",
                (summary, boundary, time.time(), session_id),
            )

    def clear_summary(self, session_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET summary = NULL, summary_boundary = NULL, "
                "updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )

    def summary(self, session_id: str) -> tuple[str, int] | None:
        row = self.connection.execute(
            "SELECT summary, summary_boundary FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None or row["summary"] is None:
            return None
        return row["summary"], row["summary_boundary"]

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

    def discard_changes_after(self, session_id: str, cursor: int) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM file_changes WHERE session_id = ? AND id > ?",
                (session_id, cursor),
            )

    def file_cursor(self, session_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(id), 0) AS cursor FROM file_changes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["cursor"])

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
