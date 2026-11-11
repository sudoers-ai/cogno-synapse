"""Ollama and Gemini reuse ONE client per instance, the way OpenAI already does (0.1.5).

Until 0.1.5 ``OllamaBackend``/``OllamaEmbedder`` built a fresh ``httpx.AsyncClient`` inside an
``async with`` on EVERY call, and ``GeminiBackend`` built a fresh ``genai.Client`` on every call.
Each one owns its connection pool, so every call paid a new TCP handshake and nothing stayed
warm — the defect 0.1.4 had already removed from OpenAI/Anthropic/Groq (``_client_cache``).

The request a provider sees does not change: same endpoint, same payload, same timeouts (the
``is_available`` probe keeps its own 3 s by overriding it per request). These proofs count how
many clients a run builds and that ``aclose`` closes the cached one — they mirror the OpenAI
reuse proofs in ``test_cloud_timeouts_retries_client.py``, with a fake client (no network).
"""

from __future__ import annotations

import asyncio
import sys
import types

import httpx
import pytest

from cogno_synapse.ollama import OllamaBackend, OllamaEmbedder


# ── Ollama: one httpx client, reused ──────────────────────────────────────────────────────

class _FakeHttpxClient:
    """A fake ``httpx.AsyncClient`` that counts how many were built and records each request."""

    built = 0

    def __init__(self, **kwargs) -> None:
        type(self).built += 1
        self.kwargs = kwargs
        self.closed = False
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[tuple[str, dict]] = []

    async def post(self, url, json=None, **kw):
        self.posts.append((url, kw))
        # One payload that satisfies both readers: the backend reads `response`/token counts,
        # the embedder reads `embeddings`/`prompt_eval_count`.
        return httpx.Response(200, json={"response": "ok", "prompt_eval_count": 1, "eval_count": 1,
                                         "embeddings": [[0.1, 0.2]]},
                              request=httpx.Request("POST", url))

    async def get(self, url, **kw):
        self.gets.append((url, kw))
        return httpx.Response(200, json={}, request=httpx.Request("GET", url))

    async def aclose(self) -> None:
        self.closed = True


def _patch_httpx(monkeypatch) -> None:
    _FakeHttpxClient.built = 0
    monkeypatch.setattr("cogno_synapse.ollama.httpx.AsyncClient", _FakeHttpxClient)


@pytest.mark.asyncio
async def test_GEMEO_n_generate_calls_build_one_ollama_client(monkeypatch):
    _patch_httpx(monkeypatch)
    b = OllamaBackend(model="m")
    for _ in range(5):
        assert (await b.generate("s", "p"))[0] == "ok"
    assert _FakeHttpxClient.built == 1                    # one client, not five
    assert b._client() is b._client()                    # the same instance every time
    await b.aclose()


@pytest.mark.asyncio
async def test_GEMEO_n_embed_calls_build_one_ollama_embedder_client(monkeypatch):
    _patch_httpx(monkeypatch)
    emb = OllamaEmbedder(model="m")
    for _ in range(5):
        await emb.embed_with_usage("olá")
    assert _FakeHttpxClient.built == 1
    assert emb._client() is emb._client()
    await emb.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_the_reused_ollama_client_and_the_next_call_rebuilds(monkeypatch):
    _patch_httpx(monkeypatch)
    b = OllamaBackend(model="m")
    await b.generate("s", "p")
    client = b._client()
    assert _FakeHttpxClient.built == 1 and client.closed is False
    await b.aclose()
    assert client.closed is True                          # LoopBoundClient called aclose()
    await b.generate("s", "p")                            # a fresh client after close
    assert _FakeHttpxClient.built == 2


@pytest.mark.asyncio
async def test_is_available_reuses_the_client_but_keeps_its_own_3s_timeout(monkeypatch):
    """Behaviour-identical to the old per-call ``httpx.AsyncClient(timeout=3)``: the probe's
    request carries ``timeout=3`` even though the pool default is the instance timeout."""
    _patch_httpx(monkeypatch)
    b = OllamaBackend(model="m")
    await b.generate("s", "p")                            # builds the shared client
    assert await b.is_available() is True
    assert _FakeHttpxClient.built == 1                    # the probe did NOT build its own
    client = b._client()
    assert client.gets[-1][0].endswith("/api/tags")
    assert client.gets[-1][1].get("timeout") == 3
    await b.aclose()


def test_a_new_event_loop_gets_a_new_ollama_client(monkeypatch):
    """An httpx pool is bound to its loop: a backend used under two ``asyncio.run``s must not
    hand the second loop the first loop's pool. Mirrors the OpenAI proof."""
    _patch_httpx(monkeypatch)
    b = OllamaBackend(model="m")

    async def grab():
        return b._client()
    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second

    async def twice():
        return b._client() is b._client()
    assert asyncio.run(twice()) is True                   # CONTROL: one loop, one client


# ── Gemini: one genai client, reused (its SDK is optional, so a fake module) ──────────────

class _FakeGenaiClient:
    built: "list[dict]" = []

    def __init__(self, **kwargs) -> None:
        type(self).built.append(kwargs)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _patch_genai(monkeypatch) -> None:
    _FakeGenaiClient.built = []
    fake_genai = types.SimpleNamespace(Client=_FakeGenaiClient)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


def test_gemini_builds_one_client_and_reuses_it(monkeypatch):
    _patch_genai(monkeypatch)
    from cogno_synapse.gemini_backend import GeminiBackend
    b = GeminiBackend(model="m", api_key="k")

    async def twice():
        return b._client() is b._client()
    assert asyncio.run(twice()) is True
    assert len(_FakeGenaiClient.built) == 1
    assert _FakeGenaiClient.built[0]["api_key"] == "k"


def test_gemini_aclose_closes_the_cached_client(monkeypatch):
    _patch_genai(monkeypatch)
    from cogno_synapse.gemini_backend import GeminiBackend
    b = GeminiBackend(model="m", api_key="k")

    async def run():
        client = b._client()
        await b.aclose()
        return client.closed
    assert asyncio.run(run()) is True


def test_gemini_a_new_event_loop_gets_a_new_client(monkeypatch):
    _patch_genai(monkeypatch)
    from cogno_synapse.gemini_backend import GeminiBackend
    b = GeminiBackend(model="m", api_key="k")

    async def grab():
        return b._client()
    assert asyncio.run(grab()) is not asyncio.run(grab())
