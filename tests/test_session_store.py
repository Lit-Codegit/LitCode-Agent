from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

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


def test_sessions_receive_stable_human_readable_aliases(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    first = store.create(tmp_path, "model", [])
    second = store.create(tmp_path, "model", [])
    aliases = {item.id: item.alias for item in store.list_sessions(tmp_path)}

    assert aliases[first] != aliases[second]
    assert aliases[first][:6].isdigit()
    assert aliases[first][6] == "-"
    assert aliases[first][11] == "-"
    assert len(aliases[first]) == 15
    assert set(aliases[first][-3:]) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")

    store.close()
    reopened = SessionStore(tmp_path / "sessions.db")
    assert reopened.session_info(first).alias == aliases[first]


def test_delete_if_pristine_only_removes_a_session_without_activity(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    pristine = store.create(
        tmp_path, "model", [{"role": "system", "content": "system"}]
    )
    used = store.create(
        tmp_path, "model", [{"role": "system", "content": "system"}]
    )
    store.enqueue_message(used, "第一条投递", workspace=tmp_path)

    assert store.delete_if_pristine(pristine)
    with pytest.raises(KeyError):
        store.session_info(pristine)
    assert not store.delete_if_pristine(used)
    assert store.session_info(used).id == used


def test_existing_database_is_backfilled_with_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    from datetime import datetime

    path = tmp_path / "sessions.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE sessions ("
        "id TEXT PRIMARY KEY, workspace TEXT NOT NULL, title TEXT NOT NULL, "
        "model TEXT NOT NULL, parent_id TEXT, messages_json TEXT NOT NULL, "
        "summary TEXT, summary_boundary INTEGER, created_at REAL NOT NULL, "
        "updated_at REAL NOT NULL)"
    )
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?)",
        ("legacy-id", str(tmp_path.resolve()), "旧会话", "model", "[]", 0.0, 0.0),
    )
    connection.commit()
    connection.close()

    store = SessionStore(path)

    info = store.session_info("legacy-id")
    expected_prefix = datetime.fromtimestamp(0.0).strftime("%y%m%d-%H%M")
    assert info.alias == expected_prefix + "-" + info.alias[-3:]
    assert len(info.alias) == 15
    assert info.origin_terminal_id is None
    assert info.origin_pane_slot is None


def test_session_catalog_prioritizes_mounted_then_current_terminal(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    historical = store.create(
        tmp_path, "model", [], origin_terminal_id="T-OLD", origin_pane_slot=1
    )
    local = store.create(
        tmp_path, "model", [], origin_terminal_id="T-NOW", origin_pane_slot=3
    )
    pane_two = store.create(
        tmp_path, "model", [], origin_terminal_id="T-OLD", origin_pane_slot=4
    )
    pane_one = store.create(
        tmp_path, "model", [], origin_terminal_id="T-NOW", origin_pane_slot=2
    )

    catalog = store.session_catalog(
        tmp_path,
        current_terminal_id="T-NOW",
        mounted={pane_two: 2, pane_one: 1},
    )

    assert [entry.info.id for entry in catalog] == [
        pane_one,
        pane_two,
        local,
        historical,
    ]
    assert [(entry.scope, entry.pane_slot) for entry in catalog] == [
        ("mounted", 1),
        ("mounted", 2),
        ("current_terminal", None),
        ("history", None),
    ]


def test_builds_bounded_session_capsule_in_same_workspace(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model-a", [{"role": "system", "content": "system"}])
    store.save_messages(
        source,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "检查登录失败"},
            {"role": "assistant", "content": "根因是 token 已过期；建议刷新 token。"},
        ],
        title="登录问题",
    )
    alias = store.session_info(source).alias

    capsule = store.session_capsule(tmp_path, alias, max_chars=48)

    assert capsule.alias == alias
    assert capsule.title == "登录问题"
    assert len(capsule.content) <= 48
    assert "登录问题" in capsule.content
    assert capsule.truncated


def test_session_alias_cannot_resolve_across_workspaces(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(first, "model", [])
    alias = store.session_info(source).alias

    with pytest.raises(KeyError):
        store.session_capsule(second, alias, max_chars=100)


def test_session_inbox_preserves_source_and_read_state(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    target = store.create(tmp_path, "model", [])
    target_alias = store.session_info(target).alias

    delivered = store.send_to_session(
        tmp_path, source, target_alias, "请验证缓存测试"
    )

    assert delivered.source_session_id == source
    assert delivered.target_session_id == target
    assert delivered.content == "请验证缓存测试"
    assert not delivered.read
    assert store.inbox(target) == (delivered,)

    store.mark_inbox_read(target, delivered.id)
    assert store.inbox(target) == ()
    assert store.inbox(target, unread_only=False)[0].read


def test_session_message_cannot_cross_workspace(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(first, "model", [])
    target = store.create(second, "model", [])

    with pytest.raises(KeyError):
        store.send_to_session(
            first, source, store.session_info(target).alias, "越界消息"
        )


def test_searches_only_relevant_bounded_session_context(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    store.save_messages(
        source,
        [
            {"role": "user", "content": "数据库迁移已完成"},
            {"role": "assistant", "content": "迁移测试通过"},
            {"role": "user", "content": "CSS 颜色待定"},
        ],
    )
    alias = store.session_info(source).alias

    excerpt = store.search_session_context(
        tmp_path, alias, "迁移", max_chars=30
    )

    assert "迁移" in excerpt
    assert "CSS" not in excerpt
    assert len(excerpt) <= 30


def test_session_store_serializes_concurrent_inbox_writes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [])
    target = store.create(tmp_path, "model", [])
    alias = store.session_info(target).alias

    with ThreadPoolExecutor(max_workers=4) as pool:
        delivered = list(
            pool.map(
                lambda number: store.send_to_session(
                    tmp_path, source, alias, f"message-{number}"
                ),
                range(40),
            )
        )

    assert len({message.id for message in delivered}) == 40
    assert len(store.inbox(target)) == 40
    queued = store.queue(target, include_finished=True)
    assert {item.content for item in queued} == {
        f"message-{number}" for number in range(40)
    }
    assert [item.sequence for item in queued] == sorted(
        item.sequence for item in queued
    )


def test_session_actor_fields_and_reference_resolution_are_persistent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sessions.db"
    store = SessionStore(database)
    parent = store.create(tmp_path, "model", [], profile="explore", paused=True)
    child = store.create_child(parent, "model", [], profile="explore")
    child_info = store.session_info(child)

    assert child_info.parent_id == parent
    assert child_info.profile == "explore"
    assert store.session_id_for_reference(tmp_path, child) == child
    assert store.session_id_for_reference(tmp_path, child_info.alias) == child
    assert store.session_tree(tmp_path)[-1][1].id == child

    store.close()
    reopened = SessionStore(database)
    assert reopened.session_info(parent).paused
    assert reopened.session_info(parent).status == "paused"


def test_session_tree_orders_by_most_recent_use(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    first = store.create(tmp_path, "model-a", [], title="较早")
    second = store.create(tmp_path, "model-a", [], title="较晚")
    time.sleep(0.01)
    store.set_session_state(first, status="idle", activity="再次使用")

    roots = [info for _, info in store.session_tree(tmp_path)]
    assert roots[0].id == first
    assert roots[1].id == second
    store.close()


def test_inbox_is_visible_across_independent_store_connections(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sessions.db"
    sender_store = SessionStore(database)
    source = sender_store.create(tmp_path, "model-a", [])
    receiver_store = SessionStore(database)
    target = receiver_store.create(tmp_path, "model-b", [])
    target_alias = receiver_store.session_info(target).alias

    sender_store.send_to_session(
        tmp_path, source, target_alias, "请检查另一个终端的结果"
    )

    messages = receiver_store.inbox(target)
    assert len(messages) == 1
    assert messages[0].source_session_id == source
    assert messages[0].content == "请检查另一个终端的结果"


def test_session_reference_snapshot_is_immutable_and_traceable(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    target = store.create(tmp_path, "model", [])
    source = store.create(tmp_path, "model", [])
    info = store.session_info(source)

    snapshot = store.record_session_reference(
        target, info.alias, info.updated_at, "第一次 capsule"
    )
    store.save_messages(source, [{"role": "assistant", "content": "后来改变"}])

    loaded = store.session_references(target)[0]
    assert loaded == snapshot
    assert loaded.source_alias == info.alias
    assert loaded.capsule == "第一次 capsule"


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


def test_user_can_reorder_and_cancel_only_queued_messages(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create(tmp_path, "model", [])
    first = store.enqueue_message(session_id, "first")
    second = store.enqueue_message(session_id, "second")
    third = store.enqueue_message(session_id, "third")

    store.reorder_queued_message(session_id, third.id, first.id)
    assert [item.content for item in store.queue(session_id)] == [
        "third",
        "first",
        "second",
    ]
    store.move_queued_message(session_id, third.id, 1)
    assert [item.content for item in store.queue(session_id)] == [
        "first",
        "third",
        "second",
    ]
    store.cancel_queued_message(second.id)
    assert [item.content for item in store.queue(session_id)] == ["first", "third"]
    with pytest.raises(ValueError, match="only a queued"):
        store.cancel_queued_message(second.id)
