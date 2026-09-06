"""The embedder must ask the endpoint that COUNTS — and survive a server that has not got it.

``OllamaEmbedder`` posted to ``/api/embeddings``, the LEGACY endpoint, whose entire response is
the vector. So ``embed_with_usage`` could only ever report 0 tokens, and it did: measured on the
live box, ``embedding_tokens`` was 0 in **134 of 134** production traces and the token ledger held
**zero** rows for the two stages that embed (``noumeno``, ``id``) across 2 370 embedding rows of
the whole history. Four fields had no writer, because the writer was asking the endpoint that
does not answer.

Measured against a live Ollama 0.20.0 on 2026-09-06::

    POST /api/embeddings  ->  {"embedding": [768 floats]}                      # no count
    POST /api/embed       ->  {"embeddings": [[768 floats]], ...,
                               "prompt_eval_count": 17}

Note the two shapes: ``embedding`` SINGULAR (one vector) against ``embeddings`` PLURAL (a list
of them). Reading the new response with the old key yields ``[]`` — a silent zero vector, and
every similarity in the pipeline becomes 0.0.
"""

from __future__ import annotations

import httpx
import pytest

from cogno_synapse.ollama import OllamaEmbedder


class _Recorder:
    """A fake ``httpx.AsyncClient`` that records the calls and replays scripted responses."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, *a, **kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        self.calls.append((url, dict(json or {})))
        status, payload = self.script.pop(0)
        return httpx.Response(status, json=payload,
                              request=httpx.Request("POST", url))


def _patch(monkeypatch, recorder):
    monkeypatch.setattr("cogno_synapse.ollama.httpx.AsyncClient", recorder)


_NEW = (200, {"model": "nomic-embed-text", "embeddings": [[0.1, 0.2, 0.3]],
              "prompt_eval_count": 17})
_LEGACY = (200, {"embedding": [0.1, 0.2, 0.3]})
_NO_ENDPOINT = (404, {"error": "not found"})


@pytest.mark.asyncio
async def test_it_asks_api_embed_and_reports_the_count(monkeypatch):
    rec = _Recorder([_NEW])
    _patch(monkeypatch, rec)

    vector, tokens = await OllamaEmbedder(model="nomic-embed-text").embed_with_usage("olá")

    url, payload = rec.calls[0]
    assert url.endswith("/api/embed"), "the legacy endpoint reports no count at all"
    assert payload == {"model": "nomic-embed-text", "input": "olá"}, \
        "/api/embed takes `input`; `prompt` is the legacy field"
    assert vector == [0.1, 0.2, 0.3]
    assert tokens == 17


@pytest.mark.asyncio
async def test_the_PLURAL_response_shape_is_unwrapped(monkeypatch):
    """``/api/embed`` accepts a LIST of inputs, so one string still comes back wrapped in a
    list of vectors. Reading it with the legacy singular key gives [] — a zero vector that
    makes every cosine 0.0 without raising anything."""
    _patch(monkeypatch, _Recorder([(200, {"embeddings": [[1.0, 0.0]], "prompt_eval_count": 4})]))
    vector, tokens = await OllamaEmbedder().embed_with_usage("x")
    assert vector == [1.0, 0.0] and tokens == 4


@pytest.mark.asyncio
async def test_an_empty_answer_does_not_raise(monkeypatch):
    _patch(monkeypatch, _Recorder([(200, {"embeddings": [], "prompt_eval_count": 0})]))
    assert await OllamaEmbedder().embed_with_usage("x") == ([], 0)


# ── an Ollama too old for /api/embed ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_server_without_the_endpoint_falls_back_instead_of_failing(monkeypatch, caplog):
    """``/api/embed`` arrived in Ollama 0.2. Older boxes exist, and an embedder must never be
    the reason a turn dies — a missing count reads as a low cost, a raised exception is a dead
    conversation. The fallback keeps the VECTOR and reports 0, which is exactly what this
    client did before the change."""
    rec = _Recorder([_NO_ENDPOINT, _LEGACY])
    _patch(monkeypatch, rec)

    with caplog.at_level("WARNING", logger="cogno_synapse.ollama"):
        vector, tokens = await OllamaEmbedder().embed_with_usage("olá")

    assert [u for u, _ in rec.calls] == [
        "http://localhost:11434/api/embed", "http://localhost:11434/api/embeddings"]
    assert rec.calls[1][1] == {"model": "nomic-embed-text", "prompt": "olá"}, \
        "the legacy endpoint takes `prompt`, not `input`"
    assert vector == [0.1, 0.2, 0.3]
    assert tokens == 0
    assert "embed_endpoint_missing" in caplog.text


@pytest.mark.asyncio
async def test_the_missing_endpoint_is_learned_ONCE(monkeypatch):
    """This client runs several times per turn. Paying a 404 round trip on every embedding is
    a latency tax on exactly the deployments that are already the slowest."""
    rec = _Recorder([_NO_ENDPOINT, _LEGACY, _LEGACY, _LEGACY])
    _patch(monkeypatch, rec)

    emb = OllamaEmbedder()
    for _ in range(3):
        assert await emb.embed_with_usage("olá") == ([0.1, 0.2, 0.3], 0)
    assert [u.rsplit("/", 1)[-1] for u, _ in rec.calls] == [
        "embed", "embeddings", "embeddings", "embeddings"]


@pytest.mark.asyncio
async def test_a_REAL_error_still_propagates(monkeypatch):
    """The fallback is for a MISSING endpoint, not for a broken deployment. A 500, or a bad
    model name (400), must raise as it always did — retrying it on the legacy endpoint would
    hide a broken box behind a silent, count-less answer."""
    _patch(monkeypatch, _Recorder([(500, {"error": "out of memory"})]))
    with pytest.raises(httpx.HTTPStatusError):
        await OllamaEmbedder().embed_with_usage("olá")

    _patch(monkeypatch, _Recorder([(400, {"error": "model not found"})]))
    emb = OllamaEmbedder()
    with pytest.raises(httpx.HTTPStatusError):
        await emb.embed_with_usage("olá")
    assert emb._legacy_only is False, "a real error must not be remembered as 'old server'"


@pytest.mark.asyncio
async def test_empty_text_costs_no_request_at_all(monkeypatch):
    rec = _Recorder([])
    _patch(monkeypatch, rec)
    assert await OllamaEmbedder().embed_with_usage("") == ([], 0)
    assert rec.calls == []


@pytest.mark.asyncio
async def test_similarity_sums_the_tokens_of_BOTH_calls(monkeypatch):
    """The ID stage's goal comparison is two embeddings. Reporting one would halve the only
    number that makes embedding usage visible at all."""
    _patch(monkeypatch, _Recorder([
        (200, {"embeddings": [[1.0, 0.0]], "prompt_eval_count": 11}),
        (200, {"embeddings": [[1.0, 0.0]], "prompt_eval_count": 6}),
    ]))
    sim, tokens = await OllamaEmbedder().similarity_with_usage("a", "b")
    assert tokens == 17
    assert sim == pytest.approx(1.0)
