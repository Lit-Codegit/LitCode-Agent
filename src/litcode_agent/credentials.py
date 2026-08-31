"""用户级 API 凭据存储；项目配置只保存凭据名称。"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Mapping


class CredentialError(ValueError):
    """凭据文件无法安全读取或内容不合法。"""


def credential_file(environ: Mapping[str, str] | None = None) -> Path:
    """返回当前用户的 LitCode 凭据文件路径。"""

    values = os.environ if environ is None else environ
    configured_home = values.get("HOME", "").strip()
    home = Path(configured_home) if configured_home else Path.home()
    if not home.is_absolute():
        raise CredentialError("HOME must be an absolute path")
    return home / ".local" / "share" / "litcode" / "auth.json"


def load_api_key(name: str, environ: Mapping[str, str] | None = None) -> str:
    """按配置中的 ``apiKeyEnv`` 名称读取用户级备用凭据。"""

    path = credential_file(environ)
    store = _read_store(path)
    credentials = store["credentials"]
    assert isinstance(credentials, dict)
    entry = credentials.get(name)
    if entry is None:
        return ""
    if not isinstance(entry, dict) or entry.get("type") != "api":
        raise CredentialError(f"invalid credential entry: {name}")
    key = entry.get("key")
    if not isinstance(key, str) or not key.strip():
        raise CredentialError(f"credential key is empty: {name}")
    return key.strip()


def save_api_key(
    name: str,
    key: str,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """原子更新一个 API 凭据，并确保目录和文件仅当前用户可访问。"""

    normalized_name = validate_credential_name(name)
    normalized_key = key.strip()
    if not normalized_key:
        raise CredentialError("API key is required")
    path = credential_file(environ)
    existing = _read_store(path)
    credentials = existing.setdefault("credentials", {})
    if not isinstance(credentials, dict):
        raise CredentialError(f"invalid credentials object: {path}")
    credentials[normalized_name] = {"type": "api", "key": normalized_key}

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.parent.chmod(0o700)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".auth-",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            os.chmod(temporary_name, 0o600)
            json.dump(existing, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        os.replace(temporary_name, path)
        temporary_name = None
        if os.name == "posix":
            path.chmod(0o600)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return path


def validate_credential_name(name: str) -> str:
    """Validate and normalize the environment-style credential identifier."""

    normalized = name.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        raise CredentialError(
            "credential name must be an environment variable name"
        )
    return normalized


def _read_store(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"version": 1, "credentials": {}}
    if path.is_symlink() or not path.is_file():
        raise CredentialError(f"credential path must be a regular file: {path}")
    if path.parent.is_symlink():
        raise CredentialError(f"credential directory must not be a symlink: {path.parent}")
    if os.name == "posix":
        if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
            raise CredentialError(
                f"credential directory permissions must be 0700: {path.parent}"
            )
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise CredentialError(
                f"credential file permissions must be 0600: {path}"
            )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CredentialError(f"cannot read credential file: {path}") from error
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise CredentialError(f"unsupported credential file format: {path}")
    if not isinstance(raw.get("credentials"), dict):
        raise CredentialError(f"invalid credentials object: {path}")
    return raw
