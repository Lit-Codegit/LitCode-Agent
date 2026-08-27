from pathlib import Path

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
