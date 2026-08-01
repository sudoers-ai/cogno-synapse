"""
Integration tests for the cloud embedders — gated on real API keys.

The unit suite mocks every SDK, so nothing there proves the request/response
SHAPES are right: a wrong field name, a payload the provider rejects, or a
``dimensions`` parameter that is silently ignored would all pass mocked and fail
in production. These make one real call each and skip without credentials.

Run locally with the relevant key exported:

    OPENAI_API_KEY=… pytest tests/integration/test_cloud_embedders_live.py
"""

import importlib.util
import os

import pytest

from cogno_synapse import CachingEmbedder, GeminiEmbedder, OpenAIEmbedder, create_embedder


def _have(env: str, sdk: str = "") -> bool:
    if sdk and importlib.util.find_spec(sdk) is None:
        return False
    return bool(os.getenv(env))


openai_only = pytest.mark.skipif(not _have("OPENAI_API_KEY", "openai"),
                                 reason="needs OPENAI_API_KEY + the openai SDK")
gemini_only = pytest.mark.skipif(not _have("GEMINI_API_KEY"),
                                 reason="needs GEMINI_API_KEY")
bedrock_only = pytest.mark.skipif(not _have("AWS_ACCESS_KEY_ID", "boto3"),
                                  reason="needs AWS credentials + boto3")


# ── OpenAI ─────────────────────────────────────────────────────────────────────

@openai_only
@pytest.mark.asyncio
async def test_openai_embed_returns_a_real_vector():
    vec, tokens = await OpenAIEmbedder().embed_with_usage("the cat sat on the mat")
    assert len(vec) == 1536                 # text-embedding-3-small's native width
    assert tokens > 0                       # the provider's own count, not a heuristic


@openai_only
@pytest.mark.asyncio
async def test_openai_honours_the_dimensions_parameter():
    """The whole migration story rests on this: if the provider ignored `dimensions`,
    a cloud embedder could not drop into a vector(768) column without a schema change."""
    vec = await OpenAIEmbedder(dimensions=768).embed("the cat sat on the mat")
    assert len(vec) == 768


@openai_only
@pytest.mark.asyncio
async def test_openai_similarity_is_semantically_ordered():
    """A live sanity check that the vectors mean something — mocks cannot show this."""
    e = OpenAIEmbedder(dimensions=768)
    near = await e.similarity("I want to book a doctor's appointment",
                              "I need to schedule a medical visit")
    far = await e.similarity("I want to book a doctor's appointment",
                             "the price of tin in 1840")
    assert near > far


@openai_only
@pytest.mark.asyncio
async def test_openai_batch_matches_single_and_keeps_order():
    e = OpenAIEmbedder(dimensions=768)
    texts = ["alpha", "beta", "gamma"]
    batched = await e.embed_batch(texts)
    assert len(batched) == 3 and all(len(v) == 768 for v in batched)
    single = await e.embed("beta")
    assert batched[1] == pytest.approx(single, abs=1e-6)   # same text → same vector, right slot


# ── Gemini ─────────────────────────────────────────────────────────────────────

@gemini_only
@pytest.mark.asyncio
async def test_gemini_embed_returns_a_real_vector():
    vec, _ = await GeminiEmbedder().embed_with_usage("the cat sat on the mat")
    assert len(vec) > 0


@gemini_only
@pytest.mark.asyncio
async def test_gemini_batch_keeps_order():
    vecs = await GeminiEmbedder().embed_batch(["alpha", "beta"])
    assert len(vecs) == 2 and all(v for v in vecs)


# ── Bedrock ────────────────────────────────────────────────────────────────────

@bedrock_only
@pytest.mark.asyncio
async def test_bedrock_titan_embed_returns_a_real_vector():
    from cogno_synapse import BedrockEmbedder
    vec, _ = await BedrockEmbedder().embed_with_usage("the cat sat on the mat")
    assert len(vec) > 0


# ── the wiring the host actually uses ──────────────────────────────────────────

@openai_only
@pytest.mark.asyncio
async def test_factory_plus_cache_wrapper_end_to_end():
    """Exactly the composition `cogno_host.embedding.build_embedder` produces — the path
    every consumer gets, rather than a bare embedder no caller ever sees."""
    e = CachingEmbedder(create_embedder("openai:text-embedding-3-small", dimensions=768))
    first = await e.embed("cache me")
    assert len(first) == 768
    assert await e.embed("cache me") == first
    assert e.usage.cache_hits == 1 and e.usage.calls == 1      # second call never left

    batched = await e.embed_batch(["cache me", "but not me"])
    assert batched[0] == first                                  # served from cache
    assert len(batched[1]) == 768
