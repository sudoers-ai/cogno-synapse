"""
cogno_synapse.factory — resolve a "provider:model" string to a backend.

A slim, infra-agnostic factory: parse the provider prefix, validate the API key
for cloud providers, instantiate the right backend. A bare string (or "ollama:")
→ Ollama.

Deliberately NOT ported from the parent: the business `_FALLBACK_MATRIX` (model
ladders) and the tenant-key contextvar — those are host concerns. A host that
wants a failover chain composes ``FallbackBackend`` itself.
"""

from __future__ import annotations

import os

from cogno_synapse.errors import MissingAPIKeyError
from cogno_synapse.base import LLMBackend

_EXTERNAL = {"openai", "anthropic", "groq", "gemini", "bedrock"}
_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "bedrock": "AWS_ACCESS_KEY_ID",
}

# OpenAI-compatible providers: identical Chat Completions API (+ native tools),
# only base_url and the key env var differ — so they all reuse OpenAIBackend
# instead of a class each. NOTE: deliberately NO "mistral" prefix — it would
# clobber Ollama's `mistral:latest` (the default local model in tests/benches).
_OPENAI_COMPATIBLE = {
    "deepseek":   ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "moonshot":   ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "kimi":       ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "grok":       ("https://api.x.ai/v1", "XAI_API_KEY"),
    "xai":        ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together":   ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks":  ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
}


def _key_env(provider: str) -> str:
    if provider in _KEY_ENV:
        return _KEY_ENV[provider]
    if provider in _OPENAI_COMPATIBLE:
        return _OPENAI_COMPATIBLE[provider][1]
    return ""


def parse_model_string(model_string: str) -> tuple[str, str]:
    """"provider:model" → (provider, model). Bare/unknown prefix → ("ollama", ...)."""
    if not model_string:
        return "ollama", "llama3.2"
    if ":" in model_string:
        prefix, rest = model_string.split(":", 1)
        prefix = prefix.lower()
        if prefix in _EXTERNAL or prefix in _OPENAI_COMPATIBLE or prefix == "ollama":
            return prefix, rest
    return "ollama", model_string


def _key_present(provider: str) -> bool:
    val = os.environ.get(_key_env(provider), "")
    return bool(val) and val.lower() != "dummy" and "sk-proj-*" not in val


def create_backend(
    model_string: str,
    *,
    api_key: str | None = None,
    base_url: str = "http://localhost:11434",
    temperature: float | None = None,
    num_ctx: int | None = 8192,
    max_tokens: int = 4096,
    timeout: int = 600,
) -> LLMBackend:
    """Instantiate a single backend for ``model_string`` (e.g. "openai:gpt-4o-mini").

    ``api_key`` overrides the env-var lookup — the host passes a per-tenant key (BYOK);
    when None the provider falls back to its env var. Raises ``MissingAPIKeyError`` if a
    cloud provider is requested with neither (fail loudly, never silently degrade).
    """
    provider, model = parse_model_string(model_string)

    if (provider in _EXTERNAL or provider in _OPENAI_COMPATIBLE) \
            and not api_key and not _key_present(provider):
        raise MissingAPIKeyError(
            f"Model '{model_string}' needs {_key_env(provider)} or an explicit api_key."
        )

    if provider in _OPENAI_COMPATIBLE:
        url, env = _OPENAI_COMPATIBLE[provider]
        from cogno_synapse.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model, api_key=api_key or os.environ.get(env), base_url=url,
                             temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    if provider == "openai":
        from cogno_synapse.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model, api_key=api_key, temperature=temperature,
                             max_tokens=max_tokens, timeout=timeout)
    if provider == "anthropic":
        from cogno_synapse.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model=model, api_key=api_key, temperature=temperature,
                                max_tokens=max_tokens, timeout=timeout)
    if provider == "groq":
        from cogno_synapse.groq_backend import GroqBackend
        return GroqBackend(model=model, api_key=api_key, temperature=temperature,
                           max_tokens=max_tokens, timeout=timeout)
    if provider == "gemini":
        from cogno_synapse.gemini_backend import GeminiBackend
        return GeminiBackend(model=model, api_key=api_key, temperature=temperature,
                             max_tokens=max_tokens, timeout=timeout)
    if provider == "bedrock":
        from cogno_synapse.bedrock_backend import BedrockBackend
        return BedrockBackend(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    from cogno_synapse.ollama import OllamaBackend
    return OllamaBackend(model=model, base_url=base_url, temperature=temperature,
                         num_ctx=num_ctx, max_tokens=max_tokens, timeout=timeout)
