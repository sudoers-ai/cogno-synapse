"""
cogno_synapse.bedrock_backend — AWS Bedrock LLM backend (Converse API).

Bedrock models (Claude, Llama, Mistral, …) via the unified Converse API, with
native tool use. boto3 is synchronous, so calls run in a thread executor.
Credentials come from the standard AWS env vars. Raises on transport/auth failure.

Optional dependency: ``pip install "cogno-anima[bedrock]"`` (or ``boto3``).
"""

from __future__ import annotations

import os
import json
import asyncio
import logging

from cogno_synapse.errors import InvalidAPIKeyError
from cogno_synapse.tool_parsing import parse_tool_calls_from_text

logger = logging.getLogger("cogno_synapse.bedrock")


def _is_auth_error(exc: Exception) -> bool:
    try:
        import botocore.exceptions
    except ImportError:
        return False
    if isinstance(exc, botocore.exceptions.ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in ("UnrecognizedClientException", "AccessDeniedException",
                        "InvalidSignatureException")
    return isinstance(exc, botocore.exceptions.NoCredentialsError)


class BedrockBackend:
    """Backend for AWS Bedrock's Converse API (env-only credentials)."""

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
        if not (self.aws_access_key and self.aws_secret_key and self.aws_region):
            logger.warning("AWS credentials not fully set — Bedrock calls may fail")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError('boto3 not installed. Run: pip install "cogno-anima[bedrock]"') from exc
        self._client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            config=Config(region_name=self.aws_region, read_timeout=self.timeout,
                          connect_timeout=15, retries={"max_attempts": 0}),
        )
        return self._client

    def _inference_config(self) -> dict:
        cfg: dict = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        return cfg

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        loop = asyncio.get_running_loop()

        def _call():
            return self._get_client().converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system=[{"text": system}] if system else [],
                inferenceConfig=self._inference_config(),
            )
        try:
            resp = await loop.run_in_executor(None, _call)
            text = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
            return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"AWS credentials rejected by Bedrock (model={self.model}): {exc}"
                ) from exc
            raise

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        loop = asyncio.get_running_loop()
        bedrock_tools = [{"toolSpec": {
            "name": t.get("function", {}).get("name", ""),
            "description": t.get("function", {}).get("description", ""),
            "inputSchema": {"json": t.get("function", {}).get("parameters", {})}}} for t in tools]
        system_text, conv = self._convert_messages(messages)

        def _call():
            kwargs: dict = {"modelId": self.model, "messages": conv,
                            "inferenceConfig": self._inference_config()}
            if system_text:
                kwargs["system"] = [{"text": system_text}]
            if bedrock_tools:
                kwargs["toolConfig"] = {"tools": bedrock_tools}
                if isinstance(tool_choice, dict) and "function" in tool_choice:
                    kwargs["toolConfig"]["toolChoice"] = {"tool": {"name": tool_choice["function"]["name"]}}
                elif tool_choice == "required":
                    kwargs["toolConfig"]["toolChoice"] = {"any": {}}
                elif tool_choice == "auto":
                    kwargs["toolConfig"]["toolChoice"] = {"auto": {}}
            return self._get_client().converse(**kwargs)
        try:
            resp = await loop.run_in_executor(None, _call)
            usage = resp.get("usage", {})
            result: dict = {"content": ""}
            tool_calls = []
            for block in resp["output"]["message"].get("content", []):
                if "text" in block:
                    result["content"] += block["text"]
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    tool_calls.append({"id": tu.get("toolUseId", ""), "type": "function",
                                       "function": {"name": tu.get("name", ""),
                                                    "arguments": json.dumps(tu.get("input", {}))}})
            if tool_calls:
                result["tool_calls"] = tool_calls
            elif result["content"]:
                rescued = parse_tool_calls_from_text(result["content"], tools)
                if rescued:
                    result["tool_calls"] = rescued
            return result, usage.get("inputTokens", 0), usage.get("outputTokens", 0)
        except Exception as exc:
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(
                    f"AWS credentials rejected by Bedrock (model={self.model}): {exc}"
                ) from exc
            raise

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        system_text = ""
        out: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_text = msg.get("content", "")
                continue
            blocks = []
            if msg.get("content") and role != "tool":
                blocks.append({"text": msg["content"]})
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                    except (json.JSONDecodeError, TypeError):
                        parsed = {}
                    blocks.append({"toolUse": {"toolUseId": tc.get("id", ""),
                                               "name": func.get("name", ""), "input": parsed}})
            if role == "tool":
                blocks.append({"toolResult": {"toolUseId": msg.get("tool_call_id", ""),
                                              "content": [{"text": msg.get("content", "")}],
                                              "status": "success"}})
                role = "user"
            if blocks:
                out.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
        return system_text, out

    def supports_native_tools(self) -> bool:
        return True
