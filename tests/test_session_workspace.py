"""Unit tests for pane-holder and sibling-snapshot formatting."""

from __future__ import annotations

import time

from litcode_agent.session_store import SessionCatalogEntry, SessionInfo
from litcode_agent.session_workspace import format_sibling_sessions


def _info(identifier: str, alias: str, **overrides: object) -> SessionInfo:
    values: dict[str, object] = {
        "id": identifier,
        "alias": alias,
        "title": alias,
        "model": "model",
        "updated_at": time.time(),
    }
    values.update(overrides)
    return SessionInfo(**values)


def test_sibling_snapshot_excludes_current_session() -> None:
    entries = (
        SessionCatalogEntry(_info("me", "main"), "mounted", 1),
        SessionCatalogEntry(_info("other", "frontend"), "mounted", 2),
    )
    text = format_sibling_sessions(entries, "me")
    assert "main" not in text
    assert "frontend" in text
    assert "pane 2" in text
    assert "<litcode_sibling_sessions>" in text


def test_sibling_snapshot_marks_background_and_queues() -> None:
    entries = (
        SessionCatalogEntry(
            _info("alone", "me", status="active", activity="正在执行", queue_size=2),
            "mounted",
            1,
        ),
        SessionCatalogEntry(
            _info(
                "cv",
                "runner",
                status="idle",
                activity="等待子会话完成",
                queue_size=2,
            ),
            "history",
            None,
        ),
    )
    text = format_sibling_sessions(entries, "alone")
    assert "后台" in text
    assert "等待子会话完成" in text
    assert "队列 2" in text


def test_sibling_snapshot_returns_empty_string_without_peers() -> None:
    entries = (SessionCatalogEntry(_info("only", "self"), "mounted", 1),)
    assert format_sibling_sessions(entries, "only") == ""
