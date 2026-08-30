from pathlib import Path

import pytest

from litcode_agent.session_store import SessionStore
from litcode_agent.tools.base import FileChange
from litcode_agent.tools.workspace import Workspace


def test_persists_sessions_checkpoints_and_forks(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / ".litcode" / "sessions.db")
    messages = [{"role": "system", "content": "system"}]
    session_id = store.create(tmp_path, "model", messages)
    messages.append({"role": "user", "content": "任务"})
    store.save_messages(session_id, messages, title="任务")
    checkpoint = store.add_checkpoint(session_id, "任务", messages)

    fork_id = store.fork(session_id, checkpoint, "model")

    assert store.load(session_id) == tuple(messages)
    assert store.load(fork_id) == tuple(messages)
    assert store.list_sessions(tmp_path)[0].parent_id == session_id


def test_rewinds_and_reapplies_agent_file_changes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create(tmp_path, "model", [])
    target = tmp_path / "note.txt"
    target.write_text("before", encoding="utf-8")
    cursor = store.file_cursor(session_id)
    target.write_text("after", encoding="utf-8")
    store.record_change(
        session_id, FileChange("note.txt", "before", "after", True)
    )

    assert store.restore_files(session_id, cursor, Workspace(tmp_path)) == 1
    assert target.read_text(encoding="utf-8") == "before"
    assert store.restore_files(
        session_id, cursor, Workspace(tmp_path), forward=True
    ) == 1
    assert target.read_text(encoding="utf-8") == "after"


def test_file_rewind_refuses_to_overwrite_a_later_user_edit(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create(tmp_path, "model", [])
    target = tmp_path / "note.txt"
    target.write_text("user edit", encoding="utf-8")
    store.record_change(
        session_id, FileChange("note.txt", "before", "agent edit", True)
    )

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        store.restore_files(session_id, 0, Workspace(tmp_path))

    assert target.read_text(encoding="utf-8") == "user edit"
