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
from cogno_synapse.openai_backend import _openai_tool_call, _provider_string, _safe_close
from cogno_synapse._obs import log_done, log_request, warn_if_retryable

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
        # Groq's response is OpenAI-shaped and carries ``system_fingerprint`` too, so the same
        # question ("did the backend under this alias change between these two calls?") is
        # answerable here. None until a call runs, reset at the top of every call so a raised
        # request leaves no stale identifier. Read via ``cogno_synapse.system_fingerprint_of``.
        self.last_system_fingerprint: str | None = None
        self.last_served_model: str | None = None
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
        log_request(logger, "groq", self.model, system, prompt)
        self.last_system_fingerprint = None
        self.last_served_model = None
        try:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(**kwargs)
            usage = resp.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            self.last_system_fingerprint = _provider_string(resp, "system_fingerprint")
            self.last_served_model = _provider_string(resp, "model")
            log_done(logger, "groq", self.model, t0, tokens_in, tokens_out)
            return (resp.choices[0].message.content or "", tokens_in, tokens_out)
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(f"GROQ_API_KEY invalid/rejected (model={self.model}): {exc}") from exc
            warn_if_retryable(logger, "groq", self.model, exc)
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
        self.last_system_fingerprint = None
        self.last_served_model = None
        try:
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            usage = resp.usage
            self.last_system_fingerprint = _provider_string(resp, "system_fingerprint")
            self.last_served_model = _provider_string(resp, "model")
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
