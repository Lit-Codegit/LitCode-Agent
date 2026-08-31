from pathlib import Path

import pytest

from litcode_agent.references import (
    ReferenceError,
    build_reference_bundle,
    list_workspace_entries,
    list_workspace_files,
)
from litcode_agent.tools.workspace import Workspace
from litcode_agent.config import ReadRoot
from litcode_agent.references import list_reference_entries
from litcode_agent.session_store import SessionStore


def test_builds_deduplicated_bounded_file_snapshots(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "示例.py").write_text("abcdef", encoding="utf-8")
    workspace = Workspace(tmp_path)

    bundle = build_reference_bundle(
        "检查 @{src/示例.py} 和 @{src/示例.py}",
        workspace,
        max_file_chars=4,
        max_total_chars=10,
    )

    assert len(bundle.references) == 1
    assert bundle.references[0].content == "abcd"
    assert bundle.references[0].truncated
    assert '<file path="src/示例.py" truncated="true">' in bundle.model_text
    assert bundle.display_text == "检查 @{src/示例.py} 和 @{src/示例.py}"


@pytest.mark.parametrize("path", ["../outside.txt", ".env", "secret.pem"])
def test_rejects_unsafe_or_sensitive_references(
    tmp_path: Path, path: str
) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "secret.pem").write_text("secret", encoding="utf-8")

    with pytest.raises(ReferenceError):
        build_reference_bundle(
            f"检查 @{{{path}}}",
            Workspace(tmp_path),
            max_file_chars=100,
            max_total_chars=100,
        )


def test_rejects_non_utf8_reference(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")

    with pytest.raises(ReferenceError, match="UTF-8"):
        build_reference_bundle(
            "检查 @{binary.bin}",
            Workspace(tmp_path),
            max_file_chars=100,
            max_total_chars=100,
        )


def test_rejects_file_content_that_looks_like_credentials(tmp_path: Path) -> None:
    (tmp_path / "credentials.txt").write_text(
        "api_key=should-not-leave-workspace", encoding="utf-8"
    )

    with pytest.raises(ReferenceError, match="凭据"):
        build_reference_bundle(
            "检查 @{credentials.txt}",
            Workspace(tmp_path),
            max_file_chars=100,
            max_total_chars=100,
        )


def test_file_index_lists_regular_files_but_not_sensitive_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("pass", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")

    paths = list_workspace_files(Workspace(tmp_path))

    assert "main.py" in paths
    assert ".env" not in paths


def test_workspace_index_includes_navigable_parent_directories(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    (nested / "main.py").write_text("pass", encoding="utf-8")

    entries = list_workspace_entries(Workspace(tmp_path))

    assert entries.files == ("src/package/main.py",)
    assert entries.directories == ("src/", "src/package/")


def test_explicit_local_root_is_searchable_despite_git_exclude(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "notes.md").write_text("私人笔记", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("local/\n", encoding="utf-8")
    root = ReadRoot("local", local.resolve(), True)

    entries = list_reference_entries(Workspace(tmp_path), (root,))
    bundle = build_reference_bundle(
        "查看 @{local/notes.md}",
        Workspace(tmp_path),
        max_file_chars=100,
        max_total_chars=100,
        read_roots=(root,),
    )

    assert "local/notes.md" in entries.files
    assert "私人笔记" in bundle.model_text


def test_read_root_must_opt_in_before_reference_is_sent(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "note.md").write_text("内容", encoding="utf-8")

    root = ReadRoot("docs", reference.resolve(), False)
    assert "docs/note.md" not in list_reference_entries(
        Workspace(tmp_path), (root,)
    ).files

    with pytest.raises(ReferenceError, match="未允许发送"):
        build_reference_bundle(
            "查看 @{docs/note.md}",
            Workspace(tmp_path),
            max_file_chars=100,
            max_total_chars=100,
            read_roots=(root,),
        )


def test_builds_bounded_untrusted_session_reference_snapshot(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    source = store.create(tmp_path, "model", [{"role": "system", "content": "system"}])
    store.save_messages(
        source,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "研究缓存失效"},
            {"role": "assistant", "content": "需要按 workspace 隔离 cache key。"},
        ],
        title="缓存调查",
    )
    alias = store.session_info(source).alias

    bundle = build_reference_bundle(
        f"采用 #{{{alias}}} 的结论",
        Workspace(tmp_path),
        max_file_chars=100,
        max_total_chars=100,
        session_store=store,
        max_session_chars=96,
    )

    assert bundle.session_references[0].alias == alias
    assert len(bundle.session_references[0].content) <= 96
    assert '<session_reference alias="' in bundle.model_text
    assert "不可信数据" in bundle.model_text
    assert bundle.display_text == f"采用 #{{{alias}}} 的结论"


def test_rejects_unknown_session_reference(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    with pytest.raises(ReferenceError, match="找不到会话"):
        build_reference_bundle(
            "采用 #{260830-1432-K7M} 的结论",
            Workspace(tmp_path),
            max_file_chars=100,
            max_total_chars=100,
            session_store=store,
            max_session_chars=100,
        )
