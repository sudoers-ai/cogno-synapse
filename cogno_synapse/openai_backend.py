"""
cogno_synapse.openai_backend — OpenAI-compatible LLM backend.

OpenAI Chat Completions API (gpt-4o, gpt-4o-mini, o-series, …). Implements
``LLMBackend`` + ``ToolCallingBackend`` (native function calling).

Adapted from the parent Cogno backend: stdlib logging (no infra logger), no
tenant contextvar (the host owns key rotation), and — per cogno-anima's
errors-propagate contract — it **raises** on transport/auth failure instead of
returning ``("", 0, 0)`` (a ``FallbackBackend`` catches and tries the next).

Optional dependency: ``pip install "cogno-anima[openai]"`` (or ``openai``).
"""

from __future__ import annotations

import os
import time
import json
import logging

from cogno_synapse.errors import InvalidAPIKeyError
from cogno_synapse.tool_parsing import parse_tool_calls_from_text

logger = logging.getLogger("cogno_synapse.openai")


def _is_auth_error(exc: Exception) -> bool:
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    return status in (401, 403)


class OpenAIBackend:
    """Backend for OpenAI's Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout: int = 120,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Point at any OpenAI-compatible endpoint (DeepSeek, Moonshot/Kimi, xAI,
        # OpenRouter, Together, Fireworks, …). None → OpenAI's default base URL.
        self.base_url = base_url
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAI calls will fail")

    @property
    def _is_o_series(self) -> bool:
        m = self.model.lower()
        return m.startswith(("o1", "o3", "o4", "gpt-5"))

    def _token_limit_kwargs(self) -> dict:
        key = "max_completion_tokens" if self._is_o_series else "max_tokens"
        return {key: self.max_tokens}

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                'openai not installed. Run: pip install "cogno-anima[openai]"'
            ) from exc
        kwargs: dict = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.AsyncOpenAI(**kwargs)

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            **self._token_limit_kwargs(),
        }
        if self.temperature is not None and not self._is_o_series:
            kwargs["temperature"] = self.temperature
        try:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(**kwargs)
            usage = resp.usage
            logger.debug("generate done elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
            return (
                resp.choices[0].message.content or "",
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"OPENAI_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            raise
        finally:
            await _safe_close(client)

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], tool_choice=None,
    ) -> tuple[dict, int, int]:
        client = self._client()
        kwargs: dict = {"model": self.model, "messages": messages, **self._token_limit_kwargs()}
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None and tools:
            kwargs["tool_choice"] = tool_choice
        if self.temperature is not None and not self._is_o_series:
            kwargs["temperature"] = self.temperature
        try:
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            usage = resp.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            result: dict = {"content": msg.content or ""}
            if msg.tool_calls:
                result["tool_calls"] = [_openai_tool_call(tc) for tc in msg.tool_calls]
            elif result["content"]:
                rescued = parse_tool_calls_from_text(result["content"], tools)
                if rescued:
                    result["tool_calls"] = rescued
            return result, tokens_in, tokens_out
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"OPENAI_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            raise
        finally:
            await _safe_close(client)

    def supports_native_tools(self) -> bool:
        return True


def _openai_tool_call(tc) -> dict:
    args = tc.function.arguments
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": args if isinstance(args, str) else json.dumps(args),
        },
    }


async def _safe_close(client) -> None:
    try:
        await client.close()
    except (TypeError, AttributeError):
        pass  # MagicMock in tests / no-op clients
