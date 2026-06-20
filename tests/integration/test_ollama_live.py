"""
Integration tests for the **default local backend** — a real Ollama instance.

The unit suite mocks every HTTP client, so the OllamaBackend/OllamaEmbedder code
that actually talks to Ollama (the out-of-the-box path for a self-hosted host) is
never exercised end to end. These tests hit a real server and assert the wire
contract: non-empty text + token counts, JSON-constrained decoding, real embedding
vectors, usage accounting, and semantic similarity ordering.

Auto-skips unless Ollama is reachable **and** the model is pulled, so the suite
stays green in CI/dev without a local server. Override the models with
``COGNO_TEST_OLLAMA_MODEL`` / ``COGNO_TEST_OLLAMA_EMBED_MODEL`` and the server with
``OLLAMA_BASE_URL``.
"""

import json
import os

import httpx
import pytest

from cogno_synapse import OllamaBackend, OllamaEmbedder
from cogno_synapse.cache import CachingEmbedder

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEN_MODEL = os.getenv("COGNO_TEST_OLLAMA_MODEL", "mistral:latest")
EMBED_MODEL = os.getenv("COGNO_TEST_OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _tags() -> set[str]:
    try:
        r = httpx.get(f"{BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()


_TAGS = _tags()


def _has(model: str) -> bool:
    """True if a model (with or without an explicit ``:tag``) is pulled."""
    base = model.split(":", 1)[0]
    return any(t == model or t.split(":", 1)[0] == base for t in _TAGS)


requires_server = pytest.mark.skipif(not _TAGS, reason="Ollama server unreachable")
requires_gen = pytest.mark.skipif(not _has(GEN_MODEL), reason=f"ollama model {GEN_MODEL} unavailable")
requires_embed = pytest.mark.skipif(
    not _has(EMBED_MODEL), reason=f"ollama model {EMBED_MODEL} unavailable"
)


# ── backend ──────────────────────────────────────────────────────────────
@requires_server
async def test_is_available_true_when_server_up():
    backend = OllamaBackend(model=GEN_MODEL, base_url=BASE_URL)
    assert await backend.is_available() is True


async def test_is_available_false_for_dead_server():
    backend = OllamaBackend(model=GEN_MODEL, base_url="http://127.0.0.1:1")
    assert await backend.is_available() is False


@requires_gen
async def test_generate_returns_text_and_token_counts():
    backend = OllamaBackend(model=GEN_MODEL, base_url=BASE_URL, temperature=0.0)
    text, tokens_in, tokens_out = await backend.generate(
        "You are terse. Answer in one short word.", "Capital of France?"
    )
    assert isinstance(text, str) and text.strip()
    assert tokens_in > 0 and tokens_out > 0
    assert "paris" in text.lower()


@requires_gen
async def test_generate_json_format_is_parseable():
    backend = OllamaBackend(model=GEN_MODEL, base_url=BASE_URL, temperature=0.0, format="json")
    text, _, _ = await backend.generate(
        'Respond ONLY with a JSON object: {"city": "<capital>", "country": "France"}.',
        "What is the capital of France?",
    )
    parsed = json.loads(text)  # JSON-constrained decoding → must parse
    assert isinstance(parsed, dict)


# ── embedder ─────────────────────────────────────────────────────────────
@requires_embed
async def test_embed_returns_float_vector():
    emb = OllamaEmbedder(model=EMBED_MODEL, base_url=BASE_URL)
    vec = await emb.embed("a cat sat on the mat")
    assert isinstance(vec, list) and len(vec) > 0
    assert all(isinstance(x, float) for x in vec)


@requires_embed
async def test_embed_with_usage_reports_tokens():
    emb = OllamaEmbedder(model=EMBED_MODEL, base_url=BASE_URL)
    vec, tokens = await emb.embed_with_usage("usage accounting check")
    assert len(vec) > 0
    # Token count is sourced from Ollama's prompt_eval_count, which the
    # /api/embeddings endpoint does NOT report for every model/build (it returns
    # 0 there). The contract is the (vector, tokens) shape with tokens >= 0; the
    # host should not rely on Ollama embedding token counts being non-zero.
    assert tokens >= 0
    assert isinstance(tokens, int)


@requires_embed
async def test_similarity_orders_related_above_unrelated():
    emb = OllamaEmbedder(model=EMBED_MODEL, base_url=BASE_URL)
    related = await emb.similarity("a dog barked loudly", "the puppy made noise")
    unrelated = await emb.similarity("a dog barked loudly", "quarterly tax accounting")
    assert -1.0 <= unrelated <= related <= 1.0
    assert related > unrelated


@requires_embed
async def test_caching_embedder_serves_second_call_from_cache():
    emb = CachingEmbedder(OllamaEmbedder(model=EMBED_MODEL, base_url=BASE_URL))
    v1 = await emb.embed("cache me")
    v2 = await emb.embed("cache me")  # identical → cache hit, no second network call
    assert v1 == v2
    usage = emb.usage  # EmbeddingUsage accounting
    assert usage.calls >= 1       # one real backend call
    assert usage.cache_hits >= 1  # the second embed was served from cache
