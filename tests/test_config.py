from pathlib import Path
import json
import os

import pytest

from litcode_agent.config import ConfigurationError, Settings


def test_loads_and_normalizes_settings(tmp_path: Path) -> None:
    settings = Settings.from_env(
        tmp_path,
        {
            "OPENAI_API_KEY": " secret ",
            "OPENAI_BASE_URL": "https://gateway.example/v1",
            "LITCODE_MODEL": "example-model",
            "LITCODE_MAX_ITERATIONS": "7",
            "LITCODE_AUTO_COMPACT_CHARS": "9000",
            "LITCODE_COMMAND_TIMEOUT_SECONDS": "2.5",
            "LITCODE_MAX_OUTPUT_CHARS": "1234",
            "LITCODE_MAX_REFERENCE_FILE_CHARS": "2048",
            "LITCODE_MAX_REFERENCE_CHARS": "8192",
            "LITCODE_MAX_SESSION_REFERENCE_CHARS": "4096",
            "LITCODE_COMMAND_POLICY": "deny",
            "LITCODE_SESSION_MESSAGE_POLICY": "allow",
            "LITCODE_SESSION_READ_POLICY": "deny",
            "LITCODE_SESSION_WAKE_POLICY": "allow",
        },
    )

    assert settings.workspace == tmp_path.resolve()
    assert settings.api_key == "secret"
    assert settings.model == "example-model"
    assert settings.max_iterations == 7
    assert settings.auto_compact_chars == 9000
    assert settings.command_timeout_seconds == 2.5
    assert settings.max_output_chars == 1234
    assert settings.max_reference_file_chars == 2048
    assert settings.max_reference_chars == 8192
    assert settings.max_session_reference_chars == 4096
    assert settings.command_policy == "deny"
    assert settings.session_message_policy == "allow"
    assert settings.session_read_policy == "deny"
    assert settings.session_send_policy == "allow"
    assert settings.session_wake_policy == "allow"


def test_uses_safe_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "example-model"},
    )

    assert settings.max_iterations == 60
    assert settings.auto_compact_chars == 200_000
    assert settings.command_timeout_seconds == 30.0
    assert settings.max_output_chars == 20_000
    assert settings.max_reference_file_chars == 32_768
    assert settings.max_reference_chars == 131_072
    assert settings.max_session_reference_chars == 4096
    assert settings.command_policy == "confirm"
    assert settings.session_message_policy == "confirm"
    assert settings.session_read_policy == "allow"
    assert settings.session_send_policy == "confirm"
    assert settings.session_wake_policy == "confirm"


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "LITCODE_MODEL"])
def test_requires_model_credentials(tmp_path: Path, missing: str) -> None:
    environ = {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "example-model"}
    del environ[missing]

    with pytest.raises(ConfigurationError, match=missing):
        Settings.from_env(tmp_path, environ)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LITCODE_MAX_ITERATIONS", "0"),
        ("LITCODE_AUTO_COMPACT_CHARS", "-1"),
        ("LITCODE_COMMAND_TIMEOUT_SECONDS", "later"),
        ("LITCODE_MAX_OUTPUT_CHARS", "-1"),
        ("LITCODE_MAX_REFERENCE_FILE_CHARS", "0"),
        ("LITCODE_MAX_REFERENCE_CHARS", "many"),
        ("LITCODE_MAX_SESSION_REFERENCE_CHARS", "0"),
        ("LITCODE_COMMAND_POLICY", "sometimes"),
        ("LITCODE_SESSION_MESSAGE_POLICY", "sometimes"),
        ("LITCODE_SESSION_READ_POLICY", "sometimes"),
        ("LITCODE_SESSION_WAKE_POLICY", "sometimes"),
    ],
)
def test_rejects_invalid_limits(tmp_path: Path, name: str, value: str) -> None:
    environ = {
        "OPENAI_API_KEY": "secret",
        "LITCODE_MODEL": "example-model",
        name: value,
    }

    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env(tmp_path, environ)


def test_safe_summary_never_contains_api_key(tmp_path: Path) -> None:
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "do-not-print", "LITCODE_MODEL": "example-model"},
    )

    summary = settings.safe_summary()

    assert "do-not-print" not in repr(summary)
    assert summary["api_key_configured"] is True


def test_loads_separate_session_read_send_and_wake_policies(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {
                    "sessionRead": "deny",
                    "sessionSend": "allow",
                    "sessionWake": "deny",
                }
            }
        )
    )

    settings = Settings.load(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "model"},
    )

    assert settings.session_read_policy == "deny"
    assert settings.session_send_policy == "allow"
    assert settings.session_wake_policy == "deny"


def test_rejects_per_file_reference_limit_above_total(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="must not exceed"):
        Settings.from_env(
            tmp_path,
            {
                "OPENAI_API_KEY": "secret",
                "LITCODE_MODEL": "example-model",
                "LITCODE_MAX_REFERENCE_FILE_CHARS": "101",
                "LITCODE_MAX_REFERENCE_CHARS": "100",
            },
        )


def test_loads_project_and_local_json_with_environment_precedence(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {
                    "primary": {
                        "model": "project-model",
                        "baseURL": "https://project.example/v1",
                        "apiKeyEnv": "CUSTOM_API_KEY",
                    }
                },
                "agent": {
                    "maxIterations": 5,
                    "autoCompactChars": 8000,
                    "maxOutputChars": 1234,
                },
            }
        )
    )
    (config_dir / "settings.local.json").write_text(
        json.dumps({"agent": {"maxIterations": 7}})
    )

    settings = Settings.load(
        tmp_path,
        {
            "CUSTOM_API_KEY": "secret",
            "LITCODE_MODEL": "environment-model",
        },
    )

    assert settings.model == "environment-model"
    assert settings.model_profile == "primary"
    assert settings.base_url == "https://project.example/v1"
    assert settings.max_iterations == 7
    assert settings.auto_compact_chars == 8000
    assert settings.max_output_chars == 1234
    assert settings.api_key_env == "CUSTOM_API_KEY"
    assert settings.config_files == (
        config_dir / "settings.json",
        config_dir / "settings.local.json",
    )


def test_loads_api_key_from_user_credential_store(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {
                    "primary": {
                        "model": "model",
                        "apiKeyEnv": "PRIMARY_KEY",
                    }
                },
            }
        )
    )
    user_home = tmp_path / "user-home"
    auth_file = user_home / ".local" / "share" / "litcode" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.parent.chmod(0o700)
    auth_file.write_text(
        json.dumps(
            {
                "version": 1,
                "credentials": {
                    "PRIMARY_KEY": {"type": "api", "key": "stored-secret"}
                },
            }
        )
    )
    auth_file.chmod(0o600)

    settings = Settings.load(tmp_path, {"HOME": str(user_home)})

    assert settings.api_key == "stored-secret"
    assert settings.safe_summary()["api_key_source"] == "user credential store"


def test_environment_api_key_overrides_user_credential_store(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {"primary": {"model": "model"}},
            }
        )
    )
    user_home = tmp_path / "user-home"
    auth_file = user_home / ".local" / "share" / "litcode" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.parent.chmod(0o700)
    auth_file.write_text(
        json.dumps(
            {
                "version": 1,
                "credentials": {
                    "OPENAI_API_KEY": {"type": "api", "key": "stored-secret"}
                },
            }
        )
    )
    auth_file.chmod(0o600)

    settings = Settings.load(
        tmp_path,
        {
            "HOME": str(user_home),
            "OPENAI_API_KEY": "environment-secret",
        },
    )

    assert settings.api_key == "environment-secret"
    assert settings.safe_summary()["api_key_source"] == "environment"


@pytest.mark.parametrize("mode", [0o644, 0o700, 0o400])
@pytest.mark.skipif(os.name != "posix", reason="POSIX permission modes")
def test_rejects_credential_file_without_exact_private_mode(
    tmp_path: Path, mode: int
) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {"primary": {"model": "model"}},
            }
        )
    )
    user_home = tmp_path / "user-home"
    auth_file = user_home / ".local" / "share" / "litcode" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.parent.chmod(0o700)
    auth_file.write_text(
        json.dumps(
            {
                "version": 1,
                "credentials": {
                    "OPENAI_API_KEY": {"type": "api", "key": "secret"}
                },
            }
        )
    )
    auth_file.chmod(mode)

    with pytest.raises(ConfigurationError, match="permissions must be 0600"):
        Settings.load(tmp_path, {"HOME": str(user_home)})


@pytest.mark.parametrize("unsafe_kind", ["malformed", "symlink"])
def test_rejects_unsafe_credential_store(
    tmp_path: Path, unsafe_kind: str
) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {"primary": {"model": "model"}},
            }
        )
    )
    user_home = tmp_path / "user-home"
    auth_file = user_home / ".local" / "share" / "litcode" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.parent.chmod(0o700)
    if unsafe_kind == "malformed":
        auth_file.write_text("not-json", encoding="utf-8")
        auth_file.chmod(0o600)
        expected = "cannot read credential file"
    else:
        target = tmp_path / "elsewhere.json"
        target.write_text('{"version": 1, "credentials": {}}', encoding="utf-8")
        target.chmod(0o600)
        auth_file.symlink_to(target)
        expected = "regular file"

    with pytest.raises(ConfigurationError, match=expected):
        Settings.load(tmp_path, {"HOME": str(user_home)})


def test_rejects_direct_api_key_in_tracked_configuration(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {
                    "primary": {
                        "model": "model",
                        "apiKey": "must-not-be-stored-here",
                    }
                }
            }
        )
    )

    with pytest.raises(ConfigurationError, match="apiKey"):
        Settings.load(tmp_path, {"OPENAI_API_KEY": "secret"})


def test_parses_claude_style_command_hooks(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {"primary": {"model": "model"}},
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "run_command|apply_patch",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo checked",
                                    "timeout": 2,
                                }
                            ],
                        }
                    ]
                },
            }
        )
    )

    settings = Settings.load(tmp_path, {"OPENAI_API_KEY": "secret"})

    group = settings.hooks.pre_tool_use[0]
    assert group.matcher == "run_command|apply_patch"
    assert group.hooks[0].command == "echo checked"
    assert group.hooks[0].timeout_seconds == 2


def test_rejects_fractional_integer_setting(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {"primary": {"model": "model"}},
                "agent": {"maxIterations": 1.5},
            }
        )
    )

    with pytest.raises(ConfigurationError, match="agent.maxIterations"):
        Settings.load(tmp_path, {"OPENAI_API_KEY": "secret"})


def test_environment_can_select_another_model_profile(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {
                    "primary": {
                        "model": "large-model",
                        "apiKeyEnv": "PRIMARY_KEY",
                    },
                    "fast": {
                        "model": "fast-model",
                        "baseURL": "https://fast.example/v1",
                        "apiKeyEnv": "FAST_KEY",
                    },
                },
            }
        )
    )

    settings = Settings.load(
        tmp_path,
        {"LITCODE_DEFAULT_MODEL": "fast", "FAST_KEY": "secret"},
    )

    assert settings.model_profile == "fast"
    assert settings.model == "fast-model"
    assert settings.base_url == "https://fast.example/v1"
    assert settings.api_key_env == "FAST_KEY"


def test_rejects_unknown_default_model_profile(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "missing",
                "models": {"primary": {"model": "model"}},
            }
        )
    )

    with pytest.raises(ConfigurationError, match="not defined"):
        Settings.load(tmp_path, {"OPENAI_API_KEY": "secret"})


def test_loads_named_read_only_roots(tmp_path: Path) -> None:
    (tmp_path / "local").mkdir()
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {"primary": {"model": "model"}},
                "permissions": {
                    "readRoots": {
                        "local": {"path": "local", "sendToModel": True}
                    }
                },
            }
        )
    )

    settings = Settings.load(tmp_path, {"OPENAI_API_KEY": "secret"})

    assert settings.read_roots[0].alias == "local"
    assert settings.read_roots[0].path == (tmp_path / "local").resolve()
    assert settings.read_roots[0].send_to_model


def test_load_tui_allows_missing_key_for_fresh_users(tmp_path: Path) -> None:
    settings = Settings.load_tui(
        tmp_path, {"HOME": str(tmp_path / "fresh-home")}
    )

    assert settings.api_key == ""
    assert settings.model == ""
    assert settings.configured is False
    assert settings.api_key_source == "none"
    summary = settings.safe_summary()
    assert summary["configured"] is False
    assert summary["api_key_configured"] is False


def test_load_tui_still_strict_for_workspace_mistakes(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ConfigurationError, match="not a directory"):
        Settings.load_tui(missing, {})


def test_user_memory_client_resumes_endpoint_when_project_has_no_models(
    tmp_path: Path, monkeypatch
) -> None:
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    from litcode_agent.credentials import save_api_key, save_last_client
    from litcode_agent.credentials import LastClient

    save_api_key("DEEPSEEK_API_KEY", "sk-memory")
    save_last_client(
        LastClient("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    )

    settings = Settings.load(tmp_path, {})

    assert settings.configured is True
    assert settings.model_profile == "user-memory"
    assert settings.api_key_env == "DEEPSEEK_API_KEY"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-chat"
    assert settings.api_key_source == "user credential store"
    assert settings.api_key == "sk-memory"


def test_project_models_config_wins_over_user_memory(
    tmp_path: Path, monkeypatch
) -> None:
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    from litcode_agent.credentials import save_api_key, save_last_client
    from litcode_agent.credentials import LastClient

    save_api_key("DEEPSEEK_API_KEY", "sk-memory")
    save_api_key("OPENAI_API_KEY", "sk-project")
    save_last_client(
        LastClient("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    )
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "primary",
                "models": {"primary": {"model": "project-model"}},
            }
        )
    )

    settings = Settings.load(tmp_path, {})

    assert settings.model_profile == "primary"
    assert settings.model == "project-model"
    assert settings.api_key_env == "OPENAI_API_KEY"
    assert settings.api_key == "sk-project"


def test_explicit_env_profile_disables_user_memory(
    tmp_path: Path, monkeypatch
) -> None:
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))
    from litcode_agent.credentials import save_api_key, save_last_client
    from litcode_agent.credentials import LastClient

    save_api_key("DEEPSEEK_API_KEY", "sk-memory")
    save_last_client(
        LastClient("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    )

    # LITCODE_DEFAULT_MODEL 显式选择时应完全忽略用户级记忆端点。
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is required"):
        Settings.load(tmp_path, {"LITCODE_DEFAULT_MODEL": "fast"})
