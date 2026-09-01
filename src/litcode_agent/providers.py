"""内置 OpenAI-compatible 供应商目录。

借鉴 OpenCode 的 Models.dev 目录概念，但只保留少数验证过的端点：
目录只提供 baseURL、凭据名和取钥入口，不定义模型行为；供应商差异仍
只在模型适配器中。自定义端点由用户在 TUI 中直接输入，不进入目录。
"""

from __future__ import annotations

from dataclasses import dataclass

from litcode_agent.credentials import validate_credential_name


class ProviderError(ValueError):
    """供应商目录条目不合法。"""


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    name: str
    base_url: str | None
    api_key_env: str
    key_url: str | None
    default_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ProviderError("provider id is required")
        if len(self.id) > 24 or not self.id.replace("-", "_").isalnum():
            raise ProviderError(f"provider id is invalid: {self.id}")
        if not self.name.strip():
            raise ProviderError(f"provider name is required: {self.id}")
        if self.base_url is not None and not self.base_url.startswith(
            ("http://", "https://")
        ):
            raise ProviderError(
                f"provider base URL must be http(s): {self.id}"
            )
        object.__setattr__(
            self, "api_key_env", validate_credential_name(self.api_key_env)
        )
        if self.key_url is not None and not self.key_url.startswith(
            ("http://", "https://")
        ):
            raise ProviderError(f"provider key URL must be http(s): {self.id}")


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        key_url="https://platform.deepseek.com/api_keys",
        default_models=("deepseek-chat", "deepseek-reasoner"),
    ),
    Provider(
        id="openai",
        name="OpenAI",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        key_url="https://platform.openai.com/api-keys",
    ),
    Provider(
        id="moonshot",
        name="Moonshot Kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        key_url="https://platform.moonshot.cn/console/api-keys",
        default_models=("kimi-k2-turbo-preview", "kimi-k2.5"),
    ),
    Provider(
        id="zai",
        name="Z.AI GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZAI_API_KEY",
        key_url="https://open.bigmodel.cn/usercenter/apikeys",
        default_models=("glm-4.6", "glm-4.5"),
    ),
    Provider(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        key_url="https://openrouter.ai/keys",
        default_models=("deepseek/deepseek-chat", "anthropic/claude-sonnet-4"),
    ),
    Provider(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        key_url="https://console.groq.com/keys",
        default_models=("llama-3.3-70b-versatile", "qwen3-coder:480b"),
    ),
    Provider(
        id="siliconflow",
        name="硅基流动",
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        key_url="https://cloud.siliconflow.cn/account/ak",
        default_models=("deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-235B-A22B"),
    ),
    Provider(
        id="ollama",
        name="本地 Ollama / LM Studio",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="LITCODE_LOCAL_API_KEY",
        key_url=None,
        default_models=(),
        # 本地上常见服务不强制鉴权，凭据名保留校验但 Key 可以填占位值。
    ),
)


def provider_by_id(provider_id: str) -> Provider | None:
    return next(
        (provider for provider in PROVIDERS if provider.id == provider_id),
        None,
    )


def ordered_providers() -> tuple[Provider, ...]:
    """Popular 优先，其余按名称排序；OpenCode 的目录排序思路。"""

    popular = {"deepseek", "openai", "openrouter", "moonshot"}
    stable: list[Provider] = []
    rest: list[Provider] = []
    for provider in PROVIDERS:
        (stable if provider.id in popular else rest).append(provider)
    return tuple(stable) + tuple(
        sorted(rest, key=lambda item: item.name)
    )
