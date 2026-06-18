"""
cogno_synapse.groq_backend — Groq LLM backend (ultra-fast OpenAI-compatible API).

Groq-hosted open models (llama-3.1-8b-instant, mixtral-8x7b, …). Implements
``LLMBackend`` + ``ToolCallingBackend``. Raises on transport/auth failure.

Optional dependency: ``pip install "cogno-anima[groq]"`` (or ``groq``).
"""

from __future__ import annotations

import os
import time
import logging

from cogno_synapse.errors import InvalidAPIKeyError
from cogno_synapse.tool_parsing import parse_tool_calls_from_text
from cogno_synapse.openai_backend import _openai_tool_call, _safe_close

logger = logging.getLogger("cogno_synapse.groq")


def _is_auth_error(exc: Exception) -> bool:
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    return status in (401, 403)


class GroqBackend:
    """Backend for Groq's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set — Groq calls will fail")

    def _client(self):
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise ImportError('groq not installed. Run: pip install "cogno-anima[groq]"') from exc
        return AsyncGroq(api_key=self.api_key, timeout=self.timeout)

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
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
                raise InvalidAPIKeyError(f"GROQ_API_KEY invalid/rejected (model={self.model}): {exc}") from exc
            raise
        finally:
            await _safe_close(client)

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        client = self._client()
        kwargs: dict = {"model": self.model, "messages": messages,
                        "tools": tools, "max_tokens": self.max_tokens}
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            usage = resp.usage
            result: dict = {"content": msg.content or ""}
            if msg.tool_calls:
                result["tool_calls"] = [_openai_tool_call(tc) for tc in msg.tool_calls]
            elif result["content"]:
                rescued = parse_tool_calls_from_text(result["content"], tools)
                if rescued:
                    result["tool_calls"] = rescued
            return result, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(f"GROQ_API_KEY invalid/rejected (model={self.model}): {exc}") from exc
            raise
        finally:
            await _safe_close(client)

    def supports_native_tools(self) -> bool:
        return True
