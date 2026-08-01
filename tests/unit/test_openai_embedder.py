"""OpenAIEmbedder + create_embedder — SDK mocked, no network."""

import pytest

from cogno_synapse import OpenAIEmbedder, OllamaEmbedder, create_embedder
from cogno_synapse.errors import MissingAPIKeyError


class _FakeEmbeddings:
    """Stands in for ``client.embeddings``; records the params it was called with."""

    def __init__(self, sink, dim=3, shuffled=False):
        self._sink = sink
        self._dim = dim
        self._shuffled = shuffled

    async def create(self, **params):
        self._sink.update(params)
        n = len(params["input"])
        # Non-parallel vectors, so a cosine over a pair is a real value rather than 1.0.
        data = [type("E", (), {"index": i,
                               "embedding": [1.0] + [0.0] * i + [1.0] * (self._dim - 1 - i)})()
                for i in range(n)]
        if self._shuffled:                     # the API documents input order; don't rely on it
            data.reverse()
        usage = type("U", (), {"prompt_tokens": 7 * n})()
        return type("R", (), {"data": data, "usage": usage})()


def _embedder(sink, shuffled=False, **kw):
    e = OpenAIEmbedder(api_key="sk-test", **kw)
    e._client = type("C", (), {"embeddings": _FakeEmbeddings(sink, shuffled=shuffled)})()
    return e


@pytest.mark.asyncio
async def test_embed_returns_vector_and_provider_token_count():
    sink = {}
    vec, tokens = await _embedder(sink).embed_with_usage("hello")
    assert len(vec) == 3 and vec[0] == 1.0
    assert tokens == 7                      # from usage.prompt_tokens, not a len//4 guess
    assert sink["input"] == ["hello"]


@pytest.mark.asyncio
async def test_dimensions_is_forwarded_when_set():
    """The store's column width is fixed — the request must carry it."""
    sink = {}
    await _embedder(sink, dimensions=768).embed("hello")
    assert sink["dimensions"] == 768


@pytest.mark.asyncio
async def test_dimensions_omitted_when_unset():
    sink = {}
    await _embedder(sink).embed("hello")
    assert "dimensions" not in sink          # let the model default rather than force one


@pytest.mark.asyncio
async def test_empty_text_never_hits_the_api():
    sink = {}
    assert await _embedder(sink).embed("") == []
    assert sink == {}


@pytest.mark.asyncio
async def test_batch_preserves_order_and_keeps_empties_in_place():
    sink = {}
    vecs = await _embedder(sink).embed_batch(["a", "", "b"])
    assert sink["input"] == ["a", "b"]       # the empty is never sent…
    assert vecs[1] == []                     # …but still occupies its slot
    assert vecs[0] and vecs[2]


@pytest.mark.asyncio
async def test_out_of_order_response_is_mapped_by_index():
    """Pairing a vector with the WRONG text corrupts recall silently, so map on the
    response's own `index` rather than trusting arrival order."""
    sink = {}
    vecs = await _embedder(sink, shuffled=True).embed_batch(["a", "b", "c"])
    straight = await _embedder({}).embed_batch(["a", "b", "c"])
    assert vecs == straight


@pytest.mark.asyncio
async def test_similarity_uses_one_request_for_the_pair():
    sink = {}
    sim = await _embedder(sink).similarity("a", "b")
    assert sink["input"] == ["a", "b"]       # one round trip, not two
    assert 0.0 < sim < 1.0                   # a real cosine, not a degenerate 1.0


# ── factory ────────────────────────────────────────────────────────────────────

def test_create_embedder_resolves_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    e = create_embedder("openai:text-embedding-3-small", dimensions=768)
    assert isinstance(e, OpenAIEmbedder)
    assert e.model == "text-embedding-3-small" and e.dimensions == 768


def test_create_embedder_defaults_to_ollama():
    e = create_embedder("nomic-embed-text")
    assert isinstance(e, OllamaEmbedder) and e.model == "nomic-embed-text"


def test_create_embedder_raises_without_key(monkeypatch):
    """Fail loudly — a silent fall back to a local embedder would write a DIFFERENT
    vector space into the same column."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        create_embedder("openai:text-embedding-3-small")


def test_create_embedder_refuses_provider_without_an_implementation(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(NotImplementedError):
        create_embedder("anthropic:whatever")
