from pathlib import Path
import json

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
            "LITCODE_COMMAND_TIMEOUT_SECONDS": "2.5",
            "LITCODE_MAX_OUTPUT_CHARS": "1234",
            "LITCODE_COMMAND_POLICY": "deny",
        },
    )

    assert settings.workspace == tmp_path.resolve()
    assert settings.api_key == "secret"
    assert settings.model == "example-model"
    assert settings.max_iterations == 7
    assert settings.command_timeout_seconds == 2.5
    assert settings.max_output_chars == 1234
    assert settings.command_policy == "deny"


def test_uses_safe_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "example-model"},
    )

    assert settings.max_iterations == 20
    assert settings.command_timeout_seconds == 30.0
    assert settings.max_output_chars == 20_000
    assert settings.command_policy == "confirm"


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
        ("LITCODE_COMMAND_TIMEOUT_SECONDS", "later"),
        ("LITCODE_MAX_OUTPUT_CHARS", "-1"),
        ("LITCODE_COMMAND_POLICY", "sometimes"),
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


def test_loads_project_and_local_json_with_environment_precedence(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "model": {
                    "name": "project-model",
                    "baseURL": "https://project.example/v1",
                    "apiKeyEnv": "CUSTOM_API_KEY",
                },
                "agent": {"maxIterations": 5, "maxOutputChars": 1234},
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
    assert settings.base_url == "https://project.example/v1"
    assert settings.max_iterations == 7
    assert settings.max_output_chars == 1234
    assert settings.api_key_env == "CUSTOM_API_KEY"
    assert settings.config_files == (
        config_dir / "settings.json",
        config_dir / "settings.local.json",
    )


def test_rejects_direct_api_key_in_tracked_configuration(tmp_path: Path) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "model": {
                    "name": "model",
                    "apiKey": "must-not-be-stored-here",
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
                "model": {"name": "model"},
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
                "model": {"name": "model"},
                "agent": {"maxIterations": 1.5},
            }
        )
    )

    with pytest.raises(ConfigurationError, match="agent.maxIterations"):
        Settings.load(tmp_path, {"OPENAI_API_KEY": "secret"})
