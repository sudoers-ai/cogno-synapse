"""Unit tests for the cloud LLM backends — mocked SDK clients, no network."""

import json
import pytest

from cogno_synapse import (
    OpenAIBackend, AnthropicBackend, GroqBackend, GeminiBackend, BedrockBackend,
    FallbackBackend, create_backend, parse_model_string,
)
from cogno_synapse.base import ToolCallingBackend, LLMBackend
from cogno_synapse.errors import InvalidAPIKeyError, MissingAPIKeyError


# ── protocol conformance ─────────────────────────────────────────────

def test_all_cloud_backends_satisfy_toolcalling_protocol():
    for B in (OpenAIBackend, AnthropicBackend, GroqBackend, GeminiBackend):
        b = B(model="m", api_key="k")
        assert isinstance(b, ToolCallingBackend)
        assert isinstance(b, LLMBackend)
        assert b.supports_native_tools() is True
    bed = BedrockBackend(model="anthropic.x")
    assert isinstance(bed, ToolCallingBackend)


# ── factory ──────────────────────────────────────────────────────────

def test_parse_model_string():
    assert parse_model_string("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert parse_model_string("qwen3:8b") == ("ollama", "qwen3:8b")
    assert parse_model_string("bare") == ("ollama", "bare")
    assert parse_model_string("") == ("ollama", "llama3.2")


def test_factory_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        create_backend("openai:gpt-4o-mini")


def test_factory_ollama_default():
    from cogno_synapse import OllamaBackend
    b = create_backend("qwen3:8b")
    assert isinstance(b, OllamaBackend) and b.model == "qwen3:8b"


def test_factory_openai_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    b = create_backend("openai:gpt-4o-mini")
    assert isinstance(b, OpenAIBackend) and b.model == "gpt-4o-mini"


# ── OpenAI-compatible providers (base_url, no new class) ──────────────

def test_parse_model_string_openai_compatible():
    assert parse_model_string("deepseek:deepseek-chat") == ("deepseek", "deepseek-chat")
    assert parse_model_string("kimi:kimi-k2") == ("kimi", "kimi-k2")
    assert parse_model_string("grok:grok-2") == ("grok", "grok-2")
    assert parse_model_string("openrouter:meta/llama") == ("openrouter", "meta/llama")
    # Regression: "mistral:latest" must stay an OLLAMA model, not a cloud provider.
    assert parse_model_string("mistral:latest") == ("ollama", "mistral:latest")


def test_factory_deepseek_uses_openai_backend_with_base_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-real-key")
    b = create_backend("deepseek:deepseek-chat")
    assert isinstance(b, OpenAIBackend)
    assert b.model == "deepseek-chat"
    assert b.base_url == "https://api.deepseek.com"
    assert b.api_key == "ds-real-key"


def test_factory_kimi_maps_to_moonshot(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key")
    b = create_backend("kimi:kimi-k2")
    assert isinstance(b, OpenAIBackend)
    assert b.base_url == "https://api.moonshot.cn/v1"


def test_factory_openai_compatible_missing_key_raises(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        create_backend("grok:grok-2")


def test_openai_backend_default_has_no_base_url():
    assert OpenAIBackend(model="gpt-4o", api_key="k").base_url is None


@pytest.mark.asyncio
async def test_openai_backend_base_url_passed_to_client(monkeypatch):
    """base_url, when set, reaches AsyncOpenAI; when None it is omitted."""
    import sys
    import types as pytypes
    captured = {}

    fake_openai = pytypes.ModuleType("openai")
    fake_openai.AsyncOpenAI = lambda **kw: captured.update(kw) or FakeOpenAIClient(
        lambda **k: _resp("ok"))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    b = OpenAIBackend(model="deepseek-chat", api_key="k", base_url="https://api.deepseek.com")
    await b.generate("s", "p")
    assert captured["base_url"] == "https://api.deepseek.com"

    captured.clear()
    b2 = OpenAIBackend(model="gpt-4o", api_key="k")   # no base_url
    await b2.generate("s", "p")
    assert "base_url" not in captured


# ── OpenAI backend with a mocked client ──────────────────────────────

class _Completions:
    def __init__(self, fn):
        self._fn = fn

    async def create(self, **kw):
        return self._fn(**kw)


class _Chat:
    def __init__(self, fn):
        self.completions = _Completions(fn)


class FakeOpenAIClient:
    def __init__(self, fn):
        self.chat = _Chat(fn)

    async def close(self):
        pass


def _resp(content="hi", tool_calls=None, ti=11, to=7):
    tc_objs = []
    for tc in (tool_calls or []):
        fn = type("F", (), {"name": tc["name"], "arguments": json.dumps(tc["args"])})
        tc_objs.append(type("TC", (), {"id": tc.get("id", "c1"), "function": fn}))
    msg = type("M", (), {"content": content, "tool_calls": tc_objs or None})
    choice = type("C", (), {"message": msg})
    usage = type("U", (), {"prompt_tokens": ti, "completion_tokens": to})
    return type("R", (), {"choices": [choice], "usage": usage})


@pytest.mark.asyncio
async def test_openai_generate(monkeypatch):
    b = OpenAIBackend(model="gpt-4o-mini", api_key="k")
    monkeypatch.setattr(b, "_client", lambda: FakeOpenAIClient(lambda **kw: _resp("hello")))
    text, ti, to = await b.generate("sys", "prompt")
    assert text == "hello" and ti == 11 and to == 7


@pytest.mark.asyncio
async def test_openai_chat_with_tools_parses_tool_calls(monkeypatch):
    b = OpenAIBackend(model="gpt-4o-mini", api_key="k")
    monkeypatch.setattr(b, "_client", lambda: FakeOpenAIClient(
        lambda **kw: _resp("", tool_calls=[{"name": "add_income", "args": {"amount": 40}}])))
    msg, ti, to = await b.chat_with_tools([{"role": "user", "content": "x"}],
                                          [{"function": {"name": "add_income"}}])
    assert msg["tool_calls"][0]["function"]["name"] == "add_income"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"amount": 40}


@pytest.mark.asyncio
async def test_openai_auth_error_raises_invalid_key(monkeypatch):
    class AuthErr(Exception):
        status_code = 401

    def boom(**kw):
        raise AuthErr("nope")
    b = OpenAIBackend(model="gpt-4o-mini", api_key="bad")
    monkeypatch.setattr(b, "_client", lambda: FakeOpenAIClient(boom))
    with pytest.raises(InvalidAPIKeyError):
        await b.generate("s", "p")


@pytest.mark.asyncio
async def test_openai_generic_error_propagates(monkeypatch):
    def boom(**kw):
        raise ValueError("transient")
    b = OpenAIBackend(model="gpt-4o-mini", api_key="k")
    monkeypatch.setattr(b, "_client", lambda: FakeOpenAIClient(boom))
    with pytest.raises(ValueError):
        await b.generate("s", "p")


def test_openai_o_series_token_kwarg():
    assert OpenAIBackend(model="o1-mini")._is_o_series is True
    assert "max_completion_tokens" in OpenAIBackend(model="o3")._token_limit_kwargs()
    assert "max_tokens" in OpenAIBackend(model="gpt-4o")._token_limit_kwargs()


@pytest.mark.asyncio
async def test_backend_raises_importerror_when_sdk_missing(monkeypatch):
    # simulate the SDK not installed
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        await OpenAIBackend(model="m", api_key="k").generate("s", "p")


# ── FallbackBackend (pure, no SDK) ───────────────────────────────────

class FakeBackend:
    def __init__(self, model="fake", result=None, exc=None, native=True):
        self.model = model
        self._result = result
        self._exc = exc
        self._native = native
        self.calls = 0

    async def generate(self, system, prompt):
        self.calls += 1
        if self._exc:
            raise self._exc
        return self._result

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        self.calls += 1
        if self._exc:
            raise self._exc
        return self._result

    def supports_native_tools(self):
        return self._native


@pytest.mark.asyncio
async def test_fallback_uses_first_success():
    a = FakeBackend("a", result=("A", 1, 1))
    b = FakeBackend("b", result=("B", 1, 1))
    fb = FallbackBackend([a, b])
    assert (await fb.generate("s", "p"))[0] == "A"
    assert b.calls == 0


@pytest.mark.asyncio
async def test_fallback_fails_over_on_error():
    a = FakeBackend("a", exc=ConnectionError("down"))
    b = FakeBackend("b", result=("B", 2, 2))
    fb = FallbackBackend([a, b])
    assert (await fb.generate("s", "p")) == ("B", 2, 2)
    assert a.calls == 1 and b.calls == 1
    assert fb.model == "b"   # last successful


@pytest.mark.asyncio
async def test_fallback_raises_last_error_when_all_fail():
    a = FakeBackend("a", exc=ConnectionError("a down"))
    b = FakeBackend("b", exc=ValueError("b down"))
    fb = FallbackBackend([a, b])
    with pytest.raises(ValueError):
        await fb.generate("s", "p")


@pytest.mark.asyncio
async def test_fallback_chat_skips_non_fc_backends():
    a = FakeBackend("a", native=False)             # skipped for tools
    b = FakeBackend("b", result=({"content": "ok"}, 1, 1))
    fb = FallbackBackend([a, b])
    msg, _, _ = await fb.chat_with_tools([], [])
    assert msg == {"content": "ok"}
    assert a.calls == 0 and b.calls == 1


def test_fallback_requires_backends():
    with pytest.raises(ValueError):
        FallbackBackend([])


def test_fallback_supports_native_tools_any():
    assert FallbackBackend([FakeBackend(native=False), FakeBackend(native=True)]).supports_native_tools()
    assert not FallbackBackend([FakeBackend(native=False)]).supports_native_tools()
