import pytest
import httpx
from cogno_synapse import OllamaBackend, OllamaEmbedder, CachingEmbedder

@pytest.mark.asyncio
async def test_ollama_backend_generate_success(monkeypatch):
    """Successful generation returns text and token counts."""
    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "response": "mocked response",
                "prompt_eval_count": 10,
                "eval_count": 5
            }
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    text, tokens_in, tokens_out = await backend.generate("system prompt", "user prompt")
    
    assert text == "mocked response"
    assert tokens_in == 10
    assert tokens_out == 5

@pytest.mark.asyncio
async def test_ollama_backend_disables_thinking_by_default(monkeypatch):
    """Reasoning models route output to a separate `thinking` field and leave
    `response` empty; the backend must send think=false so JSON lands in
    `response`."""
    captured = {}

    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": '{"ok": true}', "prompt_eval_count": 3, "eval_count": 4}
        def raise_for_status(self):
            pass

    async def mock_post(self, url, json=None, **kwargs):
        captured.update(json or {})
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="qwen3.5:4b", format="json")
    text, _, _ = await backend.generate("sys", "usr")
    assert captured.get("think") is False
    assert text == '{"ok": true}'


@pytest.mark.asyncio
async def test_ollama_backend_think_can_be_enabled(monkeypatch):
    captured = {}

    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": "x", "prompt_eval_count": 1, "eval_count": 1}
        def raise_for_status(self):
            pass

    async def mock_post(self, url, json=None, **kwargs):
        captured.update(json or {})
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    backend = OllamaBackend(model="qwen3:8b", think=True)
    await backend.generate("sys", "usr")
    assert captured.get("think") is True


@pytest.mark.asyncio
async def test_ollama_backend_falls_back_to_thinking_when_response_empty(monkeypatch):
    """If `response` is empty but `thinking` holds the answer, salvage it."""
    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": "", "thinking": '{"rewritten": "hi"}',
                    "prompt_eval_count": 2, "eval_count": 6}
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    backend = OllamaBackend(model="qwen3.5:4b", format="json")
    text, _, tokens_out = await backend.generate("sys", "usr")
    assert text == '{"rewritten": "hi"}'
    assert tokens_out == 6


@pytest.mark.asyncio
async def test_ollama_backend_connection_error(monkeypatch):
    """ConnectError must propagate to the caller."""
    async def mock_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    with pytest.raises(httpx.ConnectError, match="Connection refused"):
        await backend.generate("system prompt", "user prompt")

@pytest.mark.asyncio
async def test_ollama_backend_timeout_error(monkeypatch):
    """TimeoutException must propagate to the caller."""
    async def mock_post(*args, **kwargs):
        raise httpx.ReadTimeout("Request timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    with pytest.raises(httpx.TimeoutException):
        await backend.generate("system prompt", "user prompt")

@pytest.mark.asyncio
async def test_ollama_backend_http_error(monkeypatch):
    """HTTP error status (e.g. 500) must propagate via raise_for_status."""
    class MockResponse:
        status_code = 500
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "Internal Server Error",
                request=httpx.Request("POST", "http://test"),
                response=self,
            )
        def json(self):
            return {}

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    with pytest.raises(httpx.HTTPStatusError):
        await backend.generate("system prompt", "user prompt")

@pytest.mark.asyncio
async def test_ollama_backend_options_payload(monkeypatch):
    """Verify temperature, num_ctx, max_tokens are forwarded correctly in payload."""
    captured_payloads = []

    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": "ok", "prompt_eval_count": 1, "eval_count": 1}
        def raise_for_status(self):
            pass

    async def mock_post(client, url, *args, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="test", temperature=0.7, num_ctx=4096, max_tokens=2048)
    await backend.generate("sys", "usr")

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["model"] == "test"
    assert payload["system"] == "sys"
    assert payload["prompt"] == "usr"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.7
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["num_predict"] == 2048
    assert "format" not in payload   # not set by default


@pytest.mark.asyncio
async def test_ollama_backend_format_json(monkeypatch):
    """format='json' is forwarded to constrain Ollama output to JSON."""
    captured = []

    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": "{}", "prompt_eval_count": 1, "eval_count": 1}
        def raise_for_status(self):
            pass

    async def mock_post(client, url, *args, **kwargs):
        captured.append(kwargs.get("json", {}))
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="test", format="json")
    await backend.generate("sys", "usr")
    assert captured[0]["format"] == "json"

@pytest.mark.asyncio
async def test_ollama_backend_is_available(monkeypatch):
    """is_available returns True when Ollama responds, False when it doesn't."""
    class MockResponse:
        status_code = 200

    async def mock_get_ok(*args, **kwargs):
        return MockResponse()

    async def mock_get_fail(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get_ok)
    backend = OllamaBackend(model="test")
    assert await backend.is_available() is True

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get_fail)
    assert await backend.is_available() is False

@pytest.mark.asyncio
async def test_ollama_embedder_success(monkeypatch):
    """Successful embed returns the embedding vector (OllamaEmbedder is stateless)."""
    class MockResponse:
        status_code = 200
        def json(self):
            return {"embedding": [0.1, 0.2, 0.3]}
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    embedder = OllamaEmbedder(model="nomic-embed-text")
    vec = await embedder.embed("hello")
    assert vec == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_ollama_embedder_reports_token_usage(monkeypatch):
    """embed_with_usage surfaces Ollama's prompt_eval_count as embedding tokens."""
    class MockResponse:
        status_code = 200
        def json(self):
            return {"embedding": [0.1, 0.2], "prompt_eval_count": 7}
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    embedder = OllamaEmbedder(model="nomic-embed-text")
    vec, tokens = await embedder.embed_with_usage("hello")
    assert vec == [0.1, 0.2]
    assert tokens == 7

    # empty text → no network, no tokens
    assert await embedder.embed_with_usage("") == ([], 0)


@pytest.mark.asyncio
async def test_ollama_embedder_token_usage_defaults_zero(monkeypatch):
    """Older Ollama builds omit prompt_eval_count → tokens default to 0."""
    class MockResponse:
        status_code = 200
        def json(self):
            return {"embedding": [0.1, 0.2]}
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    embedder = OllamaEmbedder(model="nomic-embed-text")
    _, tokens = await embedder.embed_with_usage("hello")
    assert tokens == 0

@pytest.mark.asyncio
async def test_ollama_embedder_empty_text():
    """Empty text returns empty vector without making any network call."""
    embedder = OllamaEmbedder()
    vec = await embedder.embed("")
    assert vec == []

@pytest.mark.asyncio
async def test_ollama_embedder_network_error_propagates(monkeypatch):
    """Network errors during embedding must propagate to the caller."""
    async def mock_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    embedder = OllamaEmbedder(model="nomic-embed-text")
    with pytest.raises(httpx.ConnectError):
        await embedder.embed("hello")

@pytest.mark.asyncio
async def test_ollama_embedder_similarity(monkeypatch):
    """Similarity computes cosine distance between two embeddings."""
    vectors = {
        "hello": [1.0, 0.0],
        "world": [0.8, 0.6]
    }
    embedder = OllamaEmbedder()
    async def mock_embed_usage(text):
        return vectors.get(text, [0.0, 0.0]), 0

    monkeypatch.setattr(embedder, "embed_with_usage", mock_embed_usage)
    sim = await embedder.similarity("hello", "world")
    assert sim == pytest.approx(0.8)


# ── CachingEmbedder (backend-agnostic LRU + token accounting) ────────

class _FakeEmbedder:
    """Minimal usage-aware embedder double for CachingEmbedder tests."""
    model = "fake-embed"

    def __init__(self, tokens_per_call=3):
        self.calls = 0
        self.tokens_per_call = tokens_per_call

    async def embed_with_usage(self, text):
        self.calls += 1
        return [float(len(text)), 1.0], self.tokens_per_call

    async def embed(self, text):
        vec, _ = await self.embed_with_usage(text)
        return vec


@pytest.mark.asyncio
async def test_caching_embedder_caches_and_accounts_tokens():
    """CachingEmbedder serves repeats from cache (0 tokens) and tracks usage."""
    inner = _FakeEmbedder(tokens_per_call=5)
    embedder = CachingEmbedder(inner)

    vec, tokens = await embedder.embed_with_usage("hello")
    assert tokens == 5
    assert inner.calls == 1

    # cache hit → no inner call, 0 tokens
    vec2, tokens2 = await embedder.embed_with_usage("hello")
    assert vec2 == vec
    assert tokens2 == 0
    assert inner.calls == 1

    assert embedder.usage.calls == 1
    assert embedder.usage.cache_hits == 1
    assert embedder.usage.tokens == 5


@pytest.mark.asyncio
async def test_caching_embedder_similarity_with_usage():
    """similarity_with_usage sums the token cost of both embeds."""
    inner = _FakeEmbedder(tokens_per_call=4)
    embedder = CachingEmbedder(inner)
    sim, tokens = await embedder.similarity_with_usage("aaa", "bbb")
    assert 0.0 <= sim <= 1.0
    assert tokens == 8   # two fresh embeds × 4


@pytest.mark.asyncio
async def test_caching_embedder_falls_back_to_plain_embed():
    """Wrapping an embedder without embed_with_usage yields 0 tokens, still caches."""
    class _PlainEmbedder:
        model = "plain"
        def __init__(self):
            self.calls = 0
        async def embed(self, text):
            self.calls += 1
            return [1.0, 2.0]

    inner = _PlainEmbedder()
    embedder = CachingEmbedder(inner)
    vec, tokens = await embedder.embed_with_usage("x")
    assert vec == [1.0, 2.0]
    assert tokens == 0
    await embedder.embed_with_usage("x")    # cache hit
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_caching_embedder_case_insensitive():
    """'Hello' and 'hello' hit the same cache entry."""
    inner = _FakeEmbedder()
    embedder = CachingEmbedder(inner)
    v1 = await embedder.embed("Hello")
    v2 = await embedder.embed("hello")
    assert v1 == v2
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_caching_embedder_bounded_lru():
    """The cache evicts least-recently-used entries beyond cache_size."""
    inner = _FakeEmbedder()
    embedder = CachingEmbedder(inner, cache_size=2)
    await embedder.embed("a")            # cache: a
    await embedder.embed("b")            # cache: a, b
    await embedder.embed("a")            # touch a → order: b, a (cache hit)
    await embedder.embed("c")            # evicts b → cache: a, c
    assert len(embedder._cache) == 2
    assert set(embedder._cache) == {"a", "c"}
    assert inner.calls == 3              # a, b, c each fetched once

    await embedder.embed("b")            # b was evicted → refetch
    assert inner.calls == 4


@pytest.mark.asyncio
async def test_caching_embedder_disabled_cache():
    """cache_size=0 disables caching — every call hits the backend."""
    inner = _FakeEmbedder()
    embedder = CachingEmbedder(inner, cache_size=0)
    await embedder.embed("a")
    await embedder.embed("a")
    assert inner.calls == 2
    assert len(embedder._cache) == 0


@pytest.mark.asyncio
async def test_caching_embedder_mirrors_model_and_resets_usage():
    inner = _FakeEmbedder()
    embedder = CachingEmbedder(inner)
    assert embedder.model == "fake-embed"
    await embedder.embed("a")
    assert embedder.usage.tokens > 0
    embedder.reset_usage()
    assert embedder.usage.tokens == 0
    assert embedder.usage.calls == 0
