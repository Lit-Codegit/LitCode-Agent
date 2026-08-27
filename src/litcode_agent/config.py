"""Environment-based runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

CommandPolicy = Literal["confirm", "deny", "allow"]


class ConfigurationError(ValueError):
    """Raised when runtime configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings needed by an agent run."""

    workspace: Path
    api_key: str
    model: str
    base_url: str | None = None
    max_iterations: int = 20
    command_timeout_seconds: float = 30.0
    max_output_chars: int = 20_000
    command_policy: CommandPolicy = "confirm"

    @classmethod
    def from_env(
        cls,
        workspace: Path,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Load settings without reading or writing a local secrets file."""

        values = os.environ if environ is None else environ
        api_key = values.get("OPENAI_API_KEY", "").strip()
        model = values.get("LITCODE_MODEL", "").strip()
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required")
        if not model:
            raise ConfigurationError("LITCODE_MODEL is required")

        max_iterations = _positive_int(values, "LITCODE_MAX_ITERATIONS", 20)
        command_timeout = _positive_float(
            values, "LITCODE_COMMAND_TIMEOUT_SECONDS", 30.0
        )
        max_output_chars = _positive_int(
            values, "LITCODE_MAX_OUTPUT_CHARS", 20_000
        )
        command_policy = values.get("LITCODE_COMMAND_POLICY", "confirm").strip()
        if command_policy not in {"confirm", "deny", "allow"}:
            raise ConfigurationError(
                "LITCODE_COMMAND_POLICY must be confirm, deny, or allow"
            )

        resolved_workspace = workspace.expanduser().resolve()
        if not resolved_workspace.is_dir():
            raise ConfigurationError(
                f"workspace is not a directory: {resolved_workspace}"
            )

        return cls(
            workspace=resolved_workspace,
            api_key=api_key,
            model=model,
            base_url=values.get("OPENAI_BASE_URL") or None,
            max_iterations=max_iterations,
            command_timeout_seconds=command_timeout,
            max_output_chars=max_output_chars,
            command_policy=cast(CommandPolicy, command_policy),
        )

    def safe_summary(self) -> dict[str, object]:
        """Return configuration that is safe to print in logs or diagnostics."""

        return {
            "workspace": str(self.workspace),
            "model": self.model,
            "base_url": self.base_url or "provider default",
            "api_key_configured": bool(self.api_key),
            "max_iterations": self.max_iterations,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_output_chars": self.max_output_chars,
            "command_policy": self.command_policy,
        }


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw_value = values.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw_value = values.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value
