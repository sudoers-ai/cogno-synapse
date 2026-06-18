"""Unit tests for the Anthropic / Groq / Gemini / Bedrock backends.

All SDKs are mocked (no network, SDKs not required): clients are monkeypatched
in, auth errors are simulated with the attributes/exception names the backends'
``_is_auth_error`` helpers look for, and Gemini's inline-imported SDK is injected
into ``sys.modules``. Covers generate + chat_with_tools (native tool_calls and
the text-rescue branch) + message conversion + error contract.
"""

import json
import sys
import types as pytypes

import pytest

from cogno_synapse import AnthropicBackend, GroqBackend, GeminiBackend, BedrockBackend
from cogno_synapse.errors import InvalidAPIKeyError


class _Box:
    """Generic attribute bag for faking SDK response objects."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ════════════════════════ Anthropic ════════════════════════

class _AnthropicMessages:
    def __init__(self, fn):
        self._fn = fn

    async def create(self, **kw):
        return self._fn(**kw)


class FakeAnthropicClient:
    def __init__(self, fn):
        self.messages = _AnthropicMessages(fn)

    async def close(self):
        pass


def _anthropic_resp(text="hi", tool_uses=None, ti=5, to=3):
    blocks = []
    if text:
        blocks.append(_Box(type="text", text=text))
    for tu in tool_uses or []:
        blocks.append(_Box(type="tool_use", id=tu["id"], name=tu["name"], input=tu["input"]))
    return _Box(content=blocks, usage=_Box(input_tokens=ti, output_tokens=to))


@pytest.mark.asyncio
async def test_anthropic_generate(monkeypatch):
    b = AnthropicBackend(model="claude-3-haiku", api_key="k", temperature=0.0)
    monkeypatch.setattr(b, "_client", lambda: FakeAnthropicClient(lambda **kw: _anthropic_resp("hello")))
    assert await b.generate("sys", "p") == ("hello", 5, 3)


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_native(monkeypatch):
    b = AnthropicBackend(model="claude-3-haiku", api_key="k")
    resp = _anthropic_resp(text="", tool_uses=[{"id": "tu1", "name": "get_balance", "input": {"x": 1}}])
    monkeypatch.setattr(b, "_client", lambda: FakeAnthropicClient(lambda **kw: resp))
    msg, ti, to = await b.chat_with_tools(
        [{"role": "user", "content": "x"}],
        [{"function": {"name": "get_balance", "parameters": {}}}], tool_choice="required")
    assert msg["tool_calls"][0]["function"]["name"] == "get_balance"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_text_rescue(monkeypatch):
    b = AnthropicBackend(model="claude-3-haiku", api_key="k")
    leak = '<TOOL_CALL>{"tool": "get_balance", "args": {}}</TOOL_CALL>'
    monkeypatch.setattr(b, "_client", lambda: FakeAnthropicClient(lambda **kw: _anthropic_resp(leak)))
    msg, _, _ = await b.chat_with_tools(
        [{"role": "user", "content": "x"}], [{"function": {"name": "get_balance"}}])
    assert msg["tool_calls"][0]["function"]["name"] == "get_balance"


@pytest.mark.asyncio
async def test_anthropic_auth_error(monkeypatch):
    def boom(**kw):
        raise type("AuthErr", (Exception,), {"status_code": 401})("nope")
    b = AnthropicBackend(model="m", api_key="bad")
    monkeypatch.setattr(b, "_client", lambda: FakeAnthropicClient(boom))
    with pytest.raises(InvalidAPIKeyError):
        await b.generate("s", "p")


@pytest.mark.asyncio
async def test_anthropic_generic_error_propagates(monkeypatch):
    def boom(**kw):
        raise ValueError("transient")
    b = AnthropicBackend(model="m", api_key="k")
    monkeypatch.setattr(b, "_client", lambda: FakeAnthropicClient(boom))
    with pytest.raises(ValueError):
        await b.generate("s", "p")


def test_anthropic_convert_messages():
    _, conv = AnthropicBackend._convert_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "t1", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "42"},
    ])
    assert conv[0] == {"role": "user", "content": "hi"}
    assert conv[1]["content"][1]["type"] == "tool_use"          # assistant tool_use block
    assert conv[2]["content"][0]["type"] == "tool_result"       # tool result as user block


# ════════════════════════ Groq (OpenAI-shaped) ════════════════════════

class _GroqCompletions:
    def __init__(self, fn):
        self._fn = fn

    async def create(self, **kw):
        return self._fn(**kw)


class FakeGroqClient:
    def __init__(self, fn):
        self.chat = _Box(completions=_GroqCompletions(fn))

    async def close(self):
        pass


def _openai_shaped(content="hi", tool_calls=None, ti=8, to=4):
    tcs = []
    for tc in tool_calls or []:
        fn = _Box(name=tc["name"], arguments=json.dumps(tc["args"]))
        tcs.append(_Box(id=tc.get("id", "c1"), function=fn))
    msg = _Box(content=content, tool_calls=tcs or None)
    return _Box(choices=[_Box(message=msg)], usage=_Box(prompt_tokens=ti, completion_tokens=to))


@pytest.mark.asyncio
async def test_groq_generate(monkeypatch):
    b = GroqBackend(model="llama-3.1-8b-instant", api_key="k", temperature=0.0)
    monkeypatch.setattr(b, "_client", lambda: FakeGroqClient(lambda **kw: _openai_shaped("hello")))
    assert await b.generate("sys", "p") == ("hello", 8, 4)


@pytest.mark.asyncio
async def test_groq_chat_with_tools_native(monkeypatch):
    b = GroqBackend(model="m", api_key="k")
    monkeypatch.setattr(b, "_client", lambda: FakeGroqClient(
        lambda **kw: _openai_shaped("", tool_calls=[{"name": "add", "args": {"n": 2}}])))
    msg, _, _ = await b.chat_with_tools([{"role": "user", "content": "x"}],
                                        [{"function": {"name": "add"}}], tool_choice="auto")
    assert msg["tool_calls"][0]["function"]["name"] == "add"


@pytest.mark.asyncio
async def test_groq_chat_with_tools_text_rescue(monkeypatch):
    b = GroqBackend(model="m", api_key="k")
    leak = '[get_balance]'
    monkeypatch.setattr(b, "_client", lambda: FakeGroqClient(lambda **kw: _openai_shaped(leak)))
    msg, _, _ = await b.chat_with_tools([{"role": "user", "content": "x"}],
                                        [{"function": {"name": "get_balance"}}])
    assert msg["tool_calls"][0]["function"]["name"] == "get_balance"


@pytest.mark.asyncio
async def test_groq_auth_error(monkeypatch):
    def boom(**kw):
        raise type("AuthenticationError", (Exception,), {})("bad key")
    b = GroqBackend(model="m", api_key="bad")
    monkeypatch.setattr(b, "_client", lambda: FakeGroqClient(boom))
    with pytest.raises(InvalidAPIKeyError):
        await b.generate("s", "p")


# ════════════════════════ Bedrock (Converse API) ════════════════════════

class FakeBedrockClient:
    def __init__(self, fn):
        self._fn = fn

    def converse(self, **kw):
        return self._fn(**kw)


def _bedrock_resp(text="hi", tool_uses=None, ti=6, to=2):
    content = []
    if text:
        content.append({"text": text})
    for tu in tool_uses or []:
        content.append({"toolUse": {"toolUseId": tu["id"], "name": tu["name"], "input": tu["input"]}})
    return {"output": {"message": {"content": content}},
            "usage": {"inputTokens": ti, "outputTokens": to}}


@pytest.mark.asyncio
async def test_bedrock_generate(monkeypatch):
    b = BedrockBackend(model="anthropic.claude-3", temperature=0.0)
    b._client = FakeBedrockClient(lambda **kw: _bedrock_resp("hello"))
    assert await b.generate("sys", "p") == ("hello", 6, 2)


@pytest.mark.asyncio
async def test_bedrock_chat_with_tools_native(monkeypatch):
    b = BedrockBackend(model="anthropic.claude-3")
    b._client = FakeBedrockClient(
        lambda **kw: _bedrock_resp("", tool_uses=[{"id": "u1", "name": "book", "input": {"d": "fri"}}]))
    msg, ti, to = await b.chat_with_tools([{"role": "user", "content": "x"}],
                                          [{"function": {"name": "book", "parameters": {}}}],
                                          tool_choice="required")
    assert msg["tool_calls"][0]["function"]["name"] == "book"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"d": "fri"}


@pytest.mark.asyncio
async def test_bedrock_chat_with_tools_text_rescue(monkeypatch):
    b = BedrockBackend(model="anthropic.claude-3")
    leak = '{"tool": "book", "args": {"d": "fri"}}'
    b._client = FakeBedrockClient(lambda **kw: _bedrock_resp(leak))
    msg, _, _ = await b.chat_with_tools([{"role": "user", "content": "x"}],
                                        [{"function": {"name": "book"}}])
    assert msg["tool_calls"][0]["function"]["name"] == "book"


@pytest.mark.asyncio
async def test_bedrock_generic_error_propagates(monkeypatch):
    def boom(**kw):
        raise RuntimeError("throttled")
    b = BedrockBackend(model="m")
    b._client = FakeBedrockClient(boom)
    with pytest.raises(RuntimeError):
        await b.generate("s", "p")


@pytest.mark.asyncio
async def test_bedrock_auth_error_with_fake_botocore(monkeypatch):
    # Inject a minimal botocore so _is_auth_error can recognise a ClientError.
    fake_botocore = pytypes.ModuleType("botocore")
    fake_exc = pytypes.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        def __init__(self, response):
            self.response = response
            super().__init__("client error")

    class NoCredentialsError(Exception):
        pass

    fake_exc.ClientError = ClientError
    fake_exc.NoCredentialsError = NoCredentialsError
    fake_botocore.exceptions = fake_exc
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exc)

    def boom(**kw):
        raise ClientError({"Error": {"Code": "AccessDeniedException"}})
    b = BedrockBackend(model="m")
    b._client = FakeBedrockClient(boom)
    with pytest.raises(InvalidAPIKeyError):
        await b.generate("s", "p")


def test_bedrock_convert_messages():
    _, conv = BedrockBackend._convert_messages([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "42"},
    ])
    assert conv[0]["content"][0]["text"] == "hi"
    assert conv[1]["content"][0]["toolUse"]["name"] == "f"
    assert conv[2]["content"][0]["toolResult"]["toolUseId"] == "t1"


# ════════════════════════ Gemini (google-genai injected) ════════════════════════

class _GeminiModels:
    def __init__(self, fn):
        self._fn = fn

    async def generate_content(self, **kw):
        return self._fn(**kw)


class FakeGeminiClient:
    def __init__(self, fn):
        self.aio = _Box(models=_GeminiModels(fn))


def _gemini_resp(text="hi", func_calls=None, ti=4, to=2):
    parts = []
    if text:
        parts.append(_Box(text=text, function_call=None))
    for fc in func_calls or []:
        parts.append(_Box(text=None, function_call=_Box(name=fc["name"], args=fc["args"])))
    candidate = _Box(content=_Box(parts=parts))
    return _Box(text=text, candidates=[candidate],
                usage_metadata=_Box(prompt_token_count=ti, candidates_token_count=to))


@pytest.mark.asyncio
async def test_gemini_generate(monkeypatch):
    b = GeminiBackend(model="gemini-2.0-flash", api_key="k", temperature=0.0)
    monkeypatch.setattr(b, "_client", lambda: FakeGeminiClient(lambda **kw: _gemini_resp("hello")))
    assert await b.generate("sys", "p") == ("hello", 4, 2)


@pytest.mark.asyncio
async def test_gemini_auth_error_by_message(monkeypatch):
    def boom(**kw):
        raise ValueError("API key not valid. Please pass a valid API key.")
    b = GeminiBackend(model="m", api_key="bad")
    monkeypatch.setattr(b, "_client", lambda: FakeGeminiClient(boom))
    with pytest.raises(InvalidAPIKeyError):
        await b.generate("s", "p")


def _inject_fake_genai(monkeypatch, response):
    """Inject google.genai + google.genai.types so chat_with_tools imports them."""
    fake_types = pytypes.ModuleType("google.genai.types")
    for name in ("FunctionDeclaration", "Tool", "Content", "Part", "FunctionCall",
                 "FunctionResponse", "ToolConfig", "FunctionCallingConfig",
                 "GenerateContentConfig"):
        setattr(fake_types, name, _Box)
    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.types = fake_types
    fake_genai.Client = lambda **kw: FakeGeminiClient(lambda **kw2: response)
    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)


@pytest.mark.asyncio
async def test_gemini_chat_with_tools_native(monkeypatch):
    resp = _gemini_resp(text="", func_calls=[{"name": "get_balance", "args": {"k": "v"}}])
    _inject_fake_genai(monkeypatch, resp)
    b = GeminiBackend(model="gemini-2.0-flash", api_key="k")
    msg, ti, to = await b.chat_with_tools([{"role": "system", "content": "s"},
                                           {"role": "user", "content": "x"}],
                                          [{"function": {"name": "get_balance", "parameters": {}}}],
                                          tool_choice="required")
    assert msg["tool_calls"][0]["function"]["name"] == "get_balance"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"k": "v"}


def test_gemini_convert_messages():
    fake_types = pytypes.ModuleType("t")
    for name in ("Content", "Part", "FunctionCall", "FunctionResponse"):
        setattr(fake_types, name, _Box)
    system_text, contents = GeminiBackend._convert_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "42"},
    ], fake_types)
    assert system_text == "be brief"
    assert len(contents) == 4


# ════════════════════════ shared: supports_native_tools ════════════════════════

def test_all_providers_support_native_tools():
    assert AnthropicBackend(model="m", api_key="k").supports_native_tools() is True
    assert GroqBackend(model="m", api_key="k").supports_native_tools() is True
    assert GeminiBackend(model="m", api_key="k").supports_native_tools() is True
    assert BedrockBackend(model="m").supports_native_tools() is True
