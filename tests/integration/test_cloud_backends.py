"""
Integration tests for cloud LLM backends — gated on real API keys.

Each test skips unless its provider's API key (and SDK) is present, so the suite
stays green in CI/dev without cloud credentials. Run locally with the relevant
key exported (and `pip install "cogno-anima[<provider>]"`).
"""

import importlib.util
import os

import pytest

from cogno_synapse import OpenAIBackend, AnthropicBackend
from cogno_synapse.base import ToolCallingBackend

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


def _have(sdk: str, env: str) -> bool:
    return importlib.util.find_spec(sdk) is not None and bool(os.getenv(env))


@pytest.mark.skipif(not _have("openai", "OPENAI_API_KEY"), reason="no openai SDK/key")
@pytest.mark.asyncio
async def test_openai_generate_real():
    b = OpenAIBackend(model="gpt-4o-mini", temperature=0.0)
    text, ti, to = await b.generate("You are terse.", "Reply with exactly: OK")
    assert text and ti > 0 and to > 0


@pytest.mark.skipif(not _have("openai", "OPENAI_API_KEY"), reason="no openai SDK/key")
@pytest.mark.asyncio
async def test_openai_native_fc_real():
    b = OpenAIBackend(model="gpt-4o-mini", temperature=0.0)
    assert isinstance(b, ToolCallingBackend)
    msg, ti, to = await b.chat_with_tools(
        [{"role": "user", "content": "What's the weather in Paris?"}], TOOLS, tool_choice="required")
    assert msg.get("tool_calls"), f"expected a tool call, got {msg}"
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.skipif(not _have("anthropic", "ANTHROPIC_API_KEY"), reason="no anthropic SDK/key")
@pytest.mark.asyncio
async def test_anthropic_generate_real():
    b = AnthropicBackend(model="claude-3-haiku-20240307", temperature=0.0)
    text, ti, to = await b.generate("You are terse.", "Reply with exactly: OK")
    assert text and ti > 0 and to > 0


@pytest.mark.skipif(not _have("anthropic", "ANTHROPIC_API_KEY"), reason="no anthropic SDK/key")
@pytest.mark.asyncio
async def test_anthropic_native_fc_real():
    b = AnthropicBackend(model="claude-3-haiku-20240307", temperature=0.0)
    msg, ti, to = await b.chat_with_tools(
        [{"role": "user", "content": "What's the weather in Paris?"}], TOOLS, tool_choice="required")
    assert msg.get("tool_calls")
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
