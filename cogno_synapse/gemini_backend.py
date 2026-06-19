"""
cogno_synapse.gemini_backend — Google Gemini LLM backend (google-genai SDK).

Implements ``LLMBackend`` + ``ToolCallingBackend``. Converts the unified
OpenAI-format messages/tools to Gemini's ``FunctionDeclaration``/``Content``/
``Part`` shapes internally. Raises on transport/auth failure.

Optional dependency: ``pip install "cogno-anima[gemini]"`` (or ``google-genai``).
"""

from __future__ import annotations

import os
import json
import time
import logging

from cogno_synapse.errors import InvalidAPIKeyError
from cogno_synapse.tool_parsing import parse_tool_calls_from_text
from cogno_synapse._obs import log_done, log_request, warn_if_retryable

logger = logging.getLogger("cogno_synapse.gemini")


def _is_auth_error(exc: Exception) -> bool:
    if type(exc).__name__ in ("AuthenticationError", "PermissionDenied", "Unauthenticated"):
        return True
    status = (getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
              or getattr(exc, "code", None))
    if status in (401, 403, 16):  # 16 = gRPC UNAUTHENTICATED
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("api key not valid", "api_key_invalid", "permission denied"))


class GeminiBackend:
    """Backend for Google's Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set — Gemini calls will fail")

    def _client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                'google-genai not installed. Run: pip install "cogno-anima[gemini]"'
            ) from exc
        return genai.Client(api_key=self.api_key)

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        client = self._client()
        config: dict = {"max_output_tokens": self.max_tokens}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if system:
            config["system_instruction"] = system
        log_request(logger, "gemini", self.model, system, prompt)
        try:
            t0 = time.perf_counter()
            resp = await client.aio.models.generate_content(
                model=self.model, contents=prompt, config=config)
            usage = getattr(resp, "usage_metadata", None)
            tokens_in = getattr(usage, "prompt_token_count", 0) if usage else 0
            tokens_out = getattr(usage, "candidates_token_count", 0) if usage else 0
            log_done(logger, "gemini", self.model, t0, tokens_in, tokens_out)
            return (resp.text or "", tokens_in, tokens_out)
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"GEMINI_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            warn_if_retryable(logger, "gemini", self.model, exc)
            raise

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                'google-genai not installed. Run: pip install "cogno-anima[gemini]"'
            ) from exc
        client = genai.Client(api_key=self.api_key)

        decls = [types.FunctionDeclaration(
            name=t.get("function", {}).get("name", ""),
            description=t.get("function", {}).get("description", ""),
            parameters=t.get("function", {}).get("parameters", {}),
        ) for t in tools]
        gemini_tools = [types.Tool(function_declarations=decls)]

        system_text, contents = self._convert_messages(messages, types)

        config: dict = {"max_output_tokens": self.max_tokens}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if system_text:
            config["system_instruction"] = system_text

        tool_config = None
        if tool_choice == "required":
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY"))

        try:
            resp = await client.aio.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools, tool_config=tool_config, **config),
            )
            usage = getattr(resp, "usage_metadata", None)
            result: dict = {"content": ""}
            tool_calls = []
            if resp.candidates and resp.candidates[0].content:
                for i, part in enumerate(resp.candidates[0].content.parts, start=1):
                    if getattr(part, "text", None):
                        result["content"] += part.text
                    elif getattr(part, "function_call", None):
                        fc = part.function_call
                        tool_calls.append({
                            "id": f"call_{i}", "type": "function",
                            "function": {"name": fc.name,
                                         "arguments": json.dumps(dict(fc.args) if fc.args else {})},
                        })
            if tool_calls:
                result["tool_calls"] = tool_calls
            elif result["content"]:
                rescued = parse_tool_calls_from_text(result["content"], tools)
                if rescued:
                    result["tool_calls"] = rescued
            return (result,
                    getattr(usage, "prompt_token_count", 0) if usage else 0,
                    getattr(usage, "candidates_token_count", 0) if usage else 0)
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"GEMINI_API_KEY invalid/rejected (model={self.model}): {exc}"
                ) from exc
            raise

    @staticmethod
    def _convert_messages(messages, types):
        system_text = ""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_text = msg.get("content", "")
            elif role == "tool":
                contents.append(types.Content(role="user", parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name="tool_result", response={"result": msg.get("content", "")}))]))
            elif role == "assistant" and msg.get("tool_calls"):
                parts = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    parts.append(types.Part(function_call=types.FunctionCall(
                        name=func.get("name", ""), args=args)))
                contents.append(types.Content(role="model", parts=parts))
            elif role == "assistant":
                contents.append(types.Content(role="model",
                                              parts=[types.Part(text=msg.get("content", ""))]))
            else:
                contents.append(types.Content(role="user",
                                              parts=[types.Part(text=msg.get("content", ""))]))
        return system_text, contents

    def supports_native_tools(self) -> bool:
        return True
