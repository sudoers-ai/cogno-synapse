"""
cogno_synapse.anthropic_backend — Anthropic Claude LLM backend (Messages API).

Implements ``LLMBackend`` + ``ToolCallingBackend``. Converts the unified
OpenAI-format messages/tools to Anthropic's format internally (tools as
``input_schema``, tool calls as ``tool_use`` content blocks, tool results as
``tool_result`` user blocks). Raises on transport/auth failure.

Optional dependency: ``pip install "cogno-anima[anthropic]"`` (or ``anthropic``).
"""

from __future__ import annotations

import os
import json
import logging

from cogno_synapse.errors import InvalidAPIKeyError
from cogno_synapse.tool_parsing import parse_tool_calls_from_text

logger = logging.getLogger("cogno_synapse.anthropic")


def _is_auth_error(exc: Exception) -> bool:
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    return status in (401, 403)


class AnthropicBackend:
    """Backend for Anthropic's Messages API. `system` is a top-level param."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — Anthropic calls will fail")

    def _client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                'anthropic not installed. Run: pip install "cogno-anima[anthropic]"'
            ) from exc
        return anthropic.AsyncAnthropic(api_key=self.api_key, timeout=self.timeout)

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        client = self._client()
        kwargs: dict = {
            "model": self.model, "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = await client.messages.create(**kwargs)
            text = resp.content[0].text if resp.content else ""
            usage = resp.usage
            return text, usage.input_tokens if usage else 0, usage.output_tokens if usage else 0
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"ANTHROPIC_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            raise
        finally:
            await _safe_close(client)

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        client = self._client()
        anthropic_tools = [
            {"name": t.get("function", {}).get("name", ""),
             "description": t.get("function", {}).get("description", ""),
             "input_schema": t.get("function", {}).get("parameters", {"type": "object", "properties": {}})}
            for t in tools
        ]
        system_text, conv = self._convert_messages(messages)
        kwargs: dict = {"model": self.model, "messages": conv,
                        "tools": anthropic_tools, "max_tokens": self.max_tokens}
        if tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif isinstance(tool_choice, dict):
            kwargs["tool_choice"] = tool_choice
        if system_text:
            kwargs["system"] = system_text
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = await client.messages.create(**kwargs)
            usage = resp.usage
            result: dict = {"content": ""}
            tool_calls = []
            for block in resp.content:
                if block.type == "text":
                    result["content"] += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id, "type": "function",
                        "function": {"name": block.name,
                                     "arguments": json.dumps(block.input)
                                     if isinstance(block.input, dict) else str(block.input)},
                    })
            if tool_calls:
                result["tool_calls"] = tool_calls
            elif result["content"]:
                rescued = parse_tool_calls_from_text(result["content"], tools)
                if rescued:
                    result["tool_calls"] = rescued
            return result, usage.input_tokens if usage else 0, usage.output_tokens if usage else 0
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"ANTHROPIC_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            raise
        finally:
            await _safe_close(client)

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """OpenAI-format messages → (system_text, anthropic_messages)."""
        system_text = ""
        out: list[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_text = msg.get("content", "")
            elif role == "tool":
                out.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", "")}]})
            elif role == "assistant" and msg.get("tool_calls"):
                blocks = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                                   "name": func.get("name", ""), "input": args})
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role, "content": msg.get("content", "")})
        return system_text, out

    def supports_native_tools(self) -> bool:
        return True


async def _safe_close(client) -> None:
    try:
        await client.close()
    except (TypeError, AttributeError):
        pass
