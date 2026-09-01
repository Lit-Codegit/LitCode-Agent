from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from litcode_agent.credentials import (
    CredentialError,
    LastClient,
    credential_available,
    credential_file,
    load_last_client,
    save_api_key,
    save_last_client,
)
from litcode_agent.providers import (
    PROVIDERS,
    Provider,
    ProviderError,
    ordered_providers,
    provider_by_id,
)


def user_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(home))
    return home


def test_catalog_has_ordered_unique_providers() -> None:
    ids = [provider.id for provider in PROVIDERS]
    assert len(ids) == len(set(ids)) == 8
    assert all(
        provider.api_key_env.replace("_", "").isalnum()
        for provider in PROVIDERS
    )
    assert all(
        provider.base_url is None
        or provider.base_url.startswith(("http://", "https://"))
        for provider in PROVIDERS
    )
    ordered = ordered_providers()
    assert tuple(provider.id for provider in ordered) == (
        "deepseek",
        "openai",
        "moonshot",
        "openrouter",
        "groq",
        "zai",
        "ollama",
        "siliconflow",
    )
    assert provider_by_id("deepseek").name == "DeepSeek"
    assert provider_by_id("nonexistent") is None


def test_provider_rejects_invalid_ids_and_names() -> None:
    with pytest.raises(ProviderError):
        Provider("bad/id", "x", "https://a", "API_KEY", None)
    with pytest.raises(ProviderError):
        Provider("ok", "x", "ftp://a", "API_KEY", None)
    with pytest.raises(CredentialError):
        Provider("ok", "x", "https://a", "1BAD", None)


def test_last_client_roundtrip_is_non_secret_and_private(
    tmp_path: Path, monkeypatch
) -> None:
    user_home(tmp_path, monkeypatch)
    path = save_last_client(
        LastClient("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    )
    assert path == credential_file()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["lastClient"] == {
        "apiKeyEnv": "DEEPSEEK_API_KEY",
        "baseURL": "https://api.deepseek.com",
        "model": "deepseek-chat",
    }
    assert "key" not in json.dumps(stored)
    loaded = load_last_client()
    assert loaded is not None
    assert loaded.api_key_env == "DEEPSEEK_API_KEY"
    assert loaded.model == "deepseek-chat"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_last_client_detects_corrupt_or_absent(
    tmp_path: Path, monkeypatch
) -> None:
    user_home(tmp_path, monkeypatch)
    assert load_last_client() is None
    save_api_key("OPENAI_API_KEY", "sk-ok")
    assert load_last_client() is None
    path = credential_file()
    store = json.loads(path.read_text(encoding="utf-8"))
    store["lastClient"] = {"apiKeyEnv": "BAD NAME", "model": "m"}
    path.write_text(json.dumps(store), encoding="utf-8")
    assert load_last_client() is None


def test_credential_available_uses_env_then_store(
    tmp_path: Path, monkeypatch
) -> None:
    user_home(tmp_path, monkeypatch)
    assert not credential_available("OPENAI_API_KEY")
    save_api_key("OPENAI_API_KEY", "sk-was-101")
    assert credential_available("OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert credential_available("OPENAI_API_KEY")
