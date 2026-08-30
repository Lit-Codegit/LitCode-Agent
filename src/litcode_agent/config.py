"""从 Claude Code 风格 JSON 文件和环境变量加载配置。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

from litcode_agent.hooks import HookCommand, HookGroup, HookSettings

CommandPolicy = Literal["confirm", "deny", "allow"]


class ConfigurationError(ValueError):
    """配置缺失、类型错误或取值非法。"""


@dataclass(frozen=True, slots=True)
class ReadRoot:
    alias: str
    path: Path
    send_to_model: bool = False


@dataclass(frozen=True, slots=True)
class Settings:
    """一次 Agent 运行所需的完整已校验配置。"""

    workspace: Path
    api_key: str
    model: str
    model_profile: str = "environment"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_iterations: int = 20
    command_timeout_seconds: float = 30.0
    max_output_chars: int = 20_000
    max_reference_file_chars: int = 32_768
    max_reference_chars: int = 131_072
    command_policy: CommandPolicy = "confirm"
    read_roots: tuple[ReadRoot, ...] = ()
    session_database: Path | None = None
    hooks: HookSettings = HookSettings()
    config_files: tuple[Path, ...] = ()

    @classmethod
    def load(
        cls,
        workspace: Path,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """加载项目配置、本机覆盖配置，再应用环境变量覆盖。"""

        root = workspace.expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"workspace is not a directory: {root}")
        merged: dict[str, object] = {}
        loaded: list[Path] = []
        for path in (
            root / ".litcode" / "settings.json",
            root / ".litcode" / "settings.local.json",
        ):
            if path.is_file():
                merged = _deep_merge(merged, _read_json_object(path))
                loaded.append(path)
        return cls._from_values(
            root,
            os.environ if environ is None else environ,
            merged,
            tuple(loaded),
        )

    @classmethod
    def from_env(
        cls,
        workspace: Path,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """仅从环境变量加载；主要用于测试和无配置文件场景。"""

        root = workspace.expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"workspace is not a directory: {root}")
        return cls._from_values(
            root,
            os.environ if environ is None else environ,
            {},
            (),
        )

    @classmethod
    def _from_values(
        cls,
        workspace: Path,
        environ: Mapping[str, str],
        raw: Mapping[str, object],
        config_files: tuple[Path, ...],
    ) -> Settings:
        _reject_unknown_keys(
            raw,
            {
                "defaultModel",
                "models",
                "agent",
                "permissions",
                "tools",
                "hooks",
                "disableAllHooks",
            },
            "settings",
        )
        models_config = _object(raw.get("models"), "models")
        agent_config = _object(raw.get("agent"), "agent")
        permissions = _object(raw.get("permissions"), "permissions")
        tools = _object(raw.get("tools"), "tools")
        command_config = _object(tools.get("command"), "tools.command")
        configured_default = _optional_string(
            raw.get("defaultModel"), "defaultModel"
        )
        model_profile = (
            environ.get("LITCODE_DEFAULT_MODEL", "").strip()
            or configured_default
            or "environment"
        )
        if models_config:
            if model_profile not in models_config:
                raise ConfigurationError(
                    f"default model profile is not defined in models: {model_profile}"
                )
            model_config = _object(
                models_config[model_profile], f"models.{model_profile}"
            )
        else:
            if configured_default is not None:
                raise ConfigurationError(
                    "models must define the profile selected by defaultModel"
                )
            model_config = {}
        _reject_unknown_keys(
            model_config,
            {"model", "baseURL", "apiKeyEnv"},
            f"models.{model_profile}",
        )
        _reject_unknown_keys(
            agent_config,
            {
                "maxIterations",
                "maxOutputChars",
                "maxReferenceFileChars",
                "maxReferenceChars",
            },
            "agent",
        )
        _reject_unknown_keys(
            permissions, {"dangerousCommands", "readRoots"}, "permissions"
        )
        _reject_unknown_keys(tools, {"command"}, "tools")
        _reject_unknown_keys(
            command_config, {"timeoutSeconds"}, "tools.command"
        )

        api_key_env = _optional_string(
            model_config.get("apiKeyEnv"),
            f"models.{model_profile}.apiKeyEnv",
        ) or "OPENAI_API_KEY"
        api_key = environ.get(api_key_env, "").strip()
        model = (
            environ.get("LITCODE_MODEL", "").strip()
            or _optional_string(
                model_config.get("model"), f"models.{model_profile}.model"
            )
            or ""
        )
        if not api_key:
            raise ConfigurationError(f"{api_key_env} is required")
        if not model:
            raise ConfigurationError(
                "selected model profile or environment variable LITCODE_MODEL "
                "must provide a model ID"
            )

        base_url = (
            environ.get("OPENAI_BASE_URL")
            if "OPENAI_BASE_URL" in environ
            else _optional_string(
                model_config.get("baseURL"),
                f"models.{model_profile}.baseURL",
            )
        )
        max_iterations = _environment_or_positive_int(
            environ,
            "LITCODE_MAX_ITERATIONS",
            agent_config.get("maxIterations"),
            "agent.maxIterations",
            20,
        )
        max_output_chars = _environment_or_positive_int(
            environ,
            "LITCODE_MAX_OUTPUT_CHARS",
            agent_config.get("maxOutputChars"),
            "agent.maxOutputChars",
            20_000,
        )
        max_reference_file_chars = _environment_or_positive_int(
            environ,
            "LITCODE_MAX_REFERENCE_FILE_CHARS",
            agent_config.get("maxReferenceFileChars"),
            "agent.maxReferenceFileChars",
            32_768,
        )
        max_reference_chars = _environment_or_positive_int(
            environ,
            "LITCODE_MAX_REFERENCE_CHARS",
            agent_config.get("maxReferenceChars"),
            "agent.maxReferenceChars",
            131_072,
        )
        if max_reference_file_chars > max_reference_chars:
            raise ConfigurationError(
                "agent.maxReferenceFileChars must not exceed "
                "agent.maxReferenceChars"
            )
        command_timeout = _environment_or_positive_float(
            environ,
            "LITCODE_COMMAND_TIMEOUT_SECONDS",
            command_config.get("timeoutSeconds"),
            "tools.command.timeoutSeconds",
            30.0,
        )
        policy_value = (
            environ.get("LITCODE_COMMAND_POLICY")
            if "LITCODE_COMMAND_POLICY" in environ
            else permissions.get("dangerousCommands", "confirm")
        )
        if policy_value not in {"confirm", "deny", "allow"}:
            policy_name = (
                "LITCODE_COMMAND_POLICY"
                if "LITCODE_COMMAND_POLICY" in environ
                else "permissions.dangerousCommands"
            )
            raise ConfigurationError(
                f"{policy_name} must be confirm, deny, or allow"
            )
        disabled = raw.get("disableAllHooks", False)
        if not isinstance(disabled, bool):
            raise ConfigurationError("disableAllHooks must be a boolean")

        read_roots = _parse_read_roots(
            workspace, permissions.get("readRoots")
        )
        return cls(
            workspace=workspace,
            api_key=api_key,
            model=model,
            model_profile=model_profile,
            base_url=base_url or None,
            api_key_env=api_key_env,
            max_iterations=max_iterations,
            command_timeout_seconds=command_timeout,
            max_output_chars=max_output_chars,
            max_reference_file_chars=max_reference_file_chars,
            max_reference_chars=max_reference_chars,
            command_policy=cast(CommandPolicy, policy_value),
            read_roots=read_roots,
            session_database=workspace / ".litcode" / "sessions.db",
            hooks=_parse_hooks(raw.get("hooks"), disabled),
            config_files=config_files,
        )

    def safe_summary(self) -> dict[str, object]:
        """返回适合日志和诊断输出的无密钥配置。"""

        return {
            "workspace": str(self.workspace),
            "config_files": [str(path) for path in self.config_files],
            "model_profile": self.model_profile,
            "model": self.model,
            "base_url": self.base_url or "provider default",
            "api_key_env": self.api_key_env,
            "api_key_configured": bool(self.api_key),
            "max_iterations": self.max_iterations,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_output_chars": self.max_output_chars,
            "max_reference_file_chars": self.max_reference_file_chars,
            "max_reference_chars": self.max_reference_chars,
            "command_policy": self.command_policy,
            "read_roots": [
                {
                    "alias": root.alias,
                    "path": str(root.path),
                    "send_to_model": root.send_to_model,
                }
                for root in self.read_roots
            ],
            "session_database": str(self.session_database),
            "hooks_enabled": not self.hooks.disabled,
            "hook_commands": self.hooks.count,
        }


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"invalid JSON in {path}: line {error.lineno}, column {error.colno}"
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration root must be an object: {path}")
    return value


def _parse_read_roots(workspace: Path, value: object) -> tuple[ReadRoot, ...]:
    roots = _object(value, "permissions.readRoots")
    result: list[ReadRoot] = []
    for alias, raw in roots.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
            raise ConfigurationError(
                f"permissions.readRoots alias is invalid: {alias}"
            )
        config = _object(raw, f"permissions.readRoots.{alias}")
        _reject_unknown_keys(
            config, {"path", "sendToModel"}, f"permissions.readRoots.{alias}"
        )
        raw_path = _optional_string(
            config.get("path"), f"permissions.readRoots.{alias}.path"
        )
        if raw_path is None:
            raise ConfigurationError(
                f"permissions.readRoots.{alias}.path is required"
            )
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = workspace / path
        path = path.resolve()
        if not path.is_dir():
            raise ConfigurationError(
                f"permissions.readRoots.{alias}.path is not a directory: {path}"
            )
        send = config.get("sendToModel", False)
        if not isinstance(send, bool):
            raise ConfigurationError(
                f"permissions.readRoots.{alias}.sendToModel must be a boolean"
            )
        result.append(ReadRoot(alias, path, send))
    return tuple(result)


def _deep_merge(
    base: Mapping[str, object], override: Mapping[str, object]
) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(previous, value)
        else:
            merged[key] = value
    return merged


def _object(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be an object")
    return value


def _reject_unknown_keys(
    values: Mapping[str, object], allowed: set[str], name: str
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {name} setting: {unknown[0]}")


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a string")
    return value.strip() or None


def _environment_or_positive_int(
    environ: Mapping[str, str],
    environment_name: str,
    configured: object,
    configured_name: str,
    default: int,
) -> int:
    if environment_name in environ:
        value: object = environ[environment_name]
        name = environment_name
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"{name} must be an integer") from error
    elif configured is None:
        return default
    else:
        name = configured_name
        if not isinstance(configured, int) or isinstance(configured, bool):
            raise ConfigurationError(f"{name} must be an integer")
        parsed = configured
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed


def _environment_or_positive_float(
    environ: Mapping[str, str],
    environment_name: str,
    configured: object,
    configured_name: str,
    default: float,
) -> float:
    value: object = environ.get(environment_name, configured)
    name = environment_name if environment_name in environ else configured_name
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed


def _parse_hooks(value: object, disabled: bool) -> HookSettings:
    raw = _object(value, "hooks")
    events = {
        "SessionStart",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SessionEnd",
    }
    _reject_unknown_keys(raw, events, "hooks")
    return HookSettings(
        disabled=disabled,
        session_start=_parse_hook_groups(raw, "SessionStart"),
        pre_tool_use=_parse_hook_groups(raw, "PreToolUse"),
        post_tool_use=_parse_hook_groups(raw, "PostToolUse"),
        post_tool_use_failure=_parse_hook_groups(raw, "PostToolUseFailure"),
        session_end=_parse_hook_groups(raw, "SessionEnd"),
    )


def _parse_hook_groups(
    raw: Mapping[str, object], event: str
) -> tuple[HookGroup, ...]:
    groups = raw.get(event, [])
    if not isinstance(groups, list):
        raise ConfigurationError(f"hooks.{event} must be an array")
    return tuple(
        _parse_hook_group(group, f"hooks.{event}[{index}]")
        for index, group in enumerate(groups)
    )


def _parse_hook_group(value: object, name: str) -> HookGroup:
    group = _object(value, name)
    _reject_unknown_keys(group, {"matcher", "hooks"}, name)
    matcher = group.get("matcher", "")
    if not isinstance(matcher, str):
        raise ConfigurationError(f"{name}.matcher must be a string")
    try:
        re.compile(matcher)
    except re.error as error:
        raise ConfigurationError(f"invalid matcher in {name}: {error}") from error
    hooks = group.get("hooks")
    if not isinstance(hooks, list) or not hooks:
        raise ConfigurationError(f"{name}.hooks must be a non-empty array")
    return HookGroup(
        matcher=matcher,
        hooks=tuple(
            _parse_hook_command(hook, f"{name}.hooks[{index}]")
            for index, hook in enumerate(hooks)
        ),
    )


def _parse_hook_command(value: object, name: str) -> HookCommand:
    hook = _object(value, name)
    _reject_unknown_keys(hook, {"type", "command", "timeout"}, name)
    if hook.get("type") != "command":
        raise ConfigurationError(f'{name}.type must be "command"')
    command = _optional_string(hook.get("command"), f"{name}.command")
    if command is None:
        raise ConfigurationError(f"{name}.command is required")
    timeout = _environment_or_positive_float(
        {}, "", hook.get("timeout"), f"{name}.timeout", 10.0
    )
    return HookCommand(command=command, timeout_seconds=timeout)
