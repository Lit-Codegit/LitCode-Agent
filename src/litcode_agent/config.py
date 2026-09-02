"""从 Claude Code 风格 JSON 文件和环境变量加载配置。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, cast

from litcode_agent.credentials import (
    CredentialError,
    LastClient,
    load_api_key,
    load_last_client,
    validate_credential_name,
)
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
    api_key_source: str = "environment"
    configured: bool = True
    max_iterations: int = 60
    auto_compact_chars: int = 200_000
    command_timeout_seconds: float = 30.0
    max_output_chars: int = 20_000
    max_reference_file_chars: int = 32_768
    max_reference_chars: int = 131_072
    max_session_reference_chars: int = 4_096
    command_policy: CommandPolicy = "confirm"
    session_read_policy: CommandPolicy = "allow"
    session_send_policy: CommandPolicy = "confirm"
    session_wake_policy: CommandPolicy = "confirm"
    read_roots: tuple[ReadRoot, ...] = ()
    user_skill_root: Path | None = None
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
        merged, loaded = _load_project_config(root)
        settings = cls._from_values(
            root,
            os.environ if environ is None else environ,
            merged,
            loaded,
            use_credential_store=True,
        )
        values = os.environ if environ is None else environ
        return replace(settings, user_skill_root=_user_skill_root(values))

    @classmethod
    def load_tui(
        cls,
        workspace: Path,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """为 TUI 加载配置，允许还没有任何 API Key：新用户先连接再使用。

        ``run``、``doctor`` 等非交互入口仍使用严格的 :meth:`load`。
        项目 ``models`` 配置缺失时，会用用户级记忆的端点作为兜底。
        """

        root = workspace.expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"workspace is not a directory: {root}")
        merged, loaded = _load_project_config(root)
        settings = cls._from_values(
            root,
            os.environ if environ is None else environ,
            merged,
            loaded,
            use_credential_store=True,
            allow_unconfigured=True,
        )
        values = os.environ if environ is None else environ
        return replace(settings, user_skill_root=_user_skill_root(values))

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
        values = os.environ if environ is None else environ
        settings = cls._from_values(
            root,
            values,
            {},
            (),
            use_credential_store=False,
        )
        return replace(settings, user_skill_root=_user_skill_root(values))

    @classmethod
    def configured_api_key_name(
        cls,
        workspace: Path,
        environ: Mapping[str, str] | None = None,
    ) -> str:
        """返回当前模型配置档声明的凭据名称，不要求密钥已经存在。"""

        root = workspace.expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"workspace is not a directory: {root}")
        merged, _ = _load_project_config(root)
        values = os.environ if environ is None else environ
        profile, model_config = _selected_model_config(merged, values)
        return _api_key_name(model_config, profile)

    @classmethod
    def _from_values(
        cls,
        workspace: Path,
        environ: Mapping[str, str],
        raw: Mapping[str, object],
        config_files: tuple[Path, ...],
        *,
        use_credential_store: bool,
        allow_unconfigured: bool = False,
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
        agent_config = _object(raw.get("agent"), "agent")
        permissions = _object(raw.get("permissions"), "permissions")
        tools = _object(raw.get("tools"), "tools")
        command_config = _object(tools.get("command"), "tools.command")
        memory_client = (
            _user_memory_client(raw, environ)
            if use_credential_store
            else None
        )
        if memory_client is not None:
            model_profile = "user-memory"
            model_config: dict[str, object] = {
                "model": memory_client.model,
                "baseURL": memory_client.base_url,
                "apiKeyEnv": memory_client.api_key_env,
            }
        else:
            model_profile, model_config = _selected_model_config(raw, environ)
        _reject_unknown_keys(
            model_config,
            {"model", "baseURL", "apiKeyEnv"},
            f"models.{model_profile}",
        )
        _reject_unknown_keys(
            agent_config,
            {
                "maxIterations",
                "autoCompactChars",
                "maxOutputChars",
                "maxReferenceFileChars",
                "maxReferenceChars",
                "maxSessionReferenceChars",
            },
            "agent",
        )
        _reject_unknown_keys(
            permissions,
            {
                "dangerousCommands",
                "sessionMessages",
                "sessionRead",
                "sessionSend",
                "sessionWake",
                "readRoots",
            },
            "permissions",
        )
        _reject_unknown_keys(tools, {"command"}, "tools")
        _reject_unknown_keys(
            command_config, {"timeoutSeconds"}, "tools.command"
        )

        api_key_env = _api_key_name(model_config, model_profile)
        api_key = environ.get(api_key_env, "").strip()
        api_key_source = "environment"
        if not api_key and use_credential_store:
            try:
                api_key = load_api_key(api_key_env, environ)
            except CredentialError as error:
                raise ConfigurationError(str(error)) from error
            api_key_source = "user credential store"
        model = (
            environ.get("LITCODE_MODEL", "").strip()
            or _optional_string(
                model_config.get("model"), f"models.{model_profile}.model"
            )
            or ""
        )
        if not api_key:
            if allow_unconfigured:
                api_key_source = "none"
            else:
                raise ConfigurationError(f"{api_key_env} is required")
        if not model:
            if not allow_unconfigured:
                raise ConfigurationError(
                    "selected model profile or environment variable "
                    "LITCODE_MODEL must provide a model ID"
                )
        configured = bool(api_key) and bool(model)

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
            60,
        )
        auto_compact_chars = _environment_or_non_negative_int(
            environ,
            "LITCODE_AUTO_COMPACT_CHARS",
            agent_config.get("autoCompactChars"),
            "agent.autoCompactChars",
            200_000,
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
        max_session_reference_chars = _environment_or_positive_int(
            environ,
            "LITCODE_MAX_SESSION_REFERENCE_CHARS",
            agent_config.get("maxSessionReferenceChars"),
            "agent.maxSessionReferenceChars",
            4_096,
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
        session_read_policy = _permission_policy(
            environ,
            "LITCODE_SESSION_READ_POLICY",
            permissions,
            "sessionRead",
            "allow",
        )
        send_environment_name = (
            "LITCODE_SESSION_SEND_POLICY"
            if "LITCODE_SESSION_SEND_POLICY" in environ
            else "LITCODE_SESSION_MESSAGE_POLICY"
        )
        send_permission_name = (
            "sessionSend" if "sessionSend" in permissions else "sessionMessages"
        )
        session_send_policy = _permission_policy(
            environ,
            send_environment_name,
            permissions,
            send_permission_name,
            "confirm",
        )
        session_wake_policy = _permission_policy(
            environ,
            "LITCODE_SESSION_WAKE_POLICY",
            permissions,
            "sessionWake",
            "confirm",
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
            api_key_source=api_key_source,
            configured=configured,
            max_iterations=max_iterations,
            auto_compact_chars=auto_compact_chars,
            command_timeout_seconds=command_timeout,
            max_output_chars=max_output_chars,
            max_reference_file_chars=max_reference_file_chars,
            max_reference_chars=max_reference_chars,
            max_session_reference_chars=max_session_reference_chars,
            command_policy=cast(CommandPolicy, policy_value),
            session_read_policy=session_read_policy,
            session_send_policy=session_send_policy,
            session_wake_policy=session_wake_policy,
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
            "api_key_source": self.api_key_source,
            "configured": self.configured,
            "max_iterations": self.max_iterations,
            "auto_compact_chars": self.auto_compact_chars,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_output_chars": self.max_output_chars,
            "max_reference_file_chars": self.max_reference_file_chars,
            "max_reference_chars": self.max_reference_chars,
            "max_session_reference_chars": self.max_session_reference_chars,
            "command_policy": self.command_policy,
            "session_read_policy": self.session_read_policy,
            "session_send_policy": self.session_send_policy,
            "session_wake_policy": self.session_wake_policy,
            "read_roots": [
                {
                    "alias": root.alias,
                    "path": str(root.path),
                    "send_to_model": root.send_to_model,
                }
                for root in self.read_roots
            ],
            "user_skill_root": (
                str(self.user_skill_root) if self.user_skill_root is not None else None
            ),
            "session_database": str(self.session_database),
            "hooks_enabled": not self.hooks.disabled,
            "hook_commands": self.hooks.count,
        }

    @property
    def session_message_policy(self) -> CommandPolicy:
        """Backward-compatible name for the now-explicit send permission."""

        return self.session_send_policy


def _user_skill_root(environ: Mapping[str, str]) -> Path:
    home = environ.get("HOME")
    return (Path(home).expanduser() if home else Path.home()) / ".agents" / "skills"


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


def _load_project_config(
    workspace: Path,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    merged: dict[str, object] = {}
    loaded: list[Path] = []
    for path in (
        workspace / ".litcode" / "settings.json",
        workspace / ".litcode" / "settings.local.json",
    ):
        if path.is_file():
            merged = _deep_merge(merged, _read_json_object(path))
            loaded.append(path)
    return merged, tuple(loaded)


def _selected_model_config(
    raw: Mapping[str, object], environ: Mapping[str, str]
) -> tuple[str, dict[str, object]]:
    models = _object(raw.get("models"), "models")
    configured_default = _optional_string(raw.get("defaultModel"), "defaultModel")
    profile = (
        environ.get("LITCODE_DEFAULT_MODEL", "").strip()
        or configured_default
        or "environment"
    )
    if models:
        if profile not in models:
            raise ConfigurationError(
                f"default model profile is not defined in models: {profile}"
            )
        return profile, _object(models[profile], f"models.{profile}")
    if configured_default is not None:
        raise ConfigurationError(
            "models must define the profile selected by defaultModel"
        )
    return profile, {}


def _user_memory_client(
    raw: Mapping[str, object], environ: Mapping[str, str]
) -> LastClient | None:
    """项目没有 models 配置且未显式选择时，用用户级记忆的端点兜底。

    显式配置（LITCODE_DEFAULT_MODEL、defaultModel）优先于记忆；
    记忆只保存到 0600 用户级文件，不会出现在项目配置中。
    """

    if "models" in raw:
        return None
    if raw.get("defaultModel") is not None:
        return None
    if environ.get("LITCODE_DEFAULT_MODEL", "").strip():
        return None
    last = load_last_client(environ)
    if last is None or not last.model:
        return None
    return last


def _api_key_name(model_config: Mapping[str, object], profile: str) -> str:
    configured = _optional_string(
        model_config.get("apiKeyEnv"), f"models.{profile}.apiKeyEnv"
    )
    try:
        return validate_credential_name(configured or "OPENAI_API_KEY")
    except CredentialError as error:
        raise ConfigurationError(str(error)) from error


def _permission_policy(
    environ: Mapping[str, str],
    environment_name: str,
    permissions: Mapping[str, object],
    permission_name: str,
    default: CommandPolicy,
) -> CommandPolicy:
    value = (
        environ.get(environment_name)
        if environment_name in environ
        else permissions.get(permission_name, default)
    )
    if value not in {"confirm", "deny", "allow"}:
        source = (
            environment_name
            if environment_name in environ
            else f"permissions.{permission_name}"
        )
        raise ConfigurationError(f"{source} must be confirm, deny, or allow")
    return cast(CommandPolicy, value)


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


def _environment_or_non_negative_int(
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
    if parsed < 0:
        raise ConfigurationError(f"{name} must be non-negative")
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
