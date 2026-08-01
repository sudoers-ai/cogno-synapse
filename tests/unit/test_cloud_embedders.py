"""GeminiEmbedder + BedrockEmbedder + create_embedder routing — all mocked, no network."""

import json

import httpx
import pytest

from cogno_synapse import BedrockEmbedder, GeminiEmbedder, OllamaEmbedder, create_embedder
from cogno_synapse.errors import InvalidAPIKeyError, MissingAPIKeyError


# ── Gemini (httpx REST) ────────────────────────────────────────────────────────

def _mock_post(monkeypatch, payload, status=200, sink=None):
    class R:
        status_code = status

        def json(self):
            return payload

        def raise_for_status(self):
            if status >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

    async def post(self, url, json=None, **kw):
        if sink is not None:
            sink["url"] = url
            sink["body"] = json
        return R()

    monkeypatch.setattr(httpx.AsyncClient, "post", post)


@pytest.mark.asyncio
async def test_gemini_embed_reads_the_values_field(monkeypatch):
    sink = {}
    _mock_post(monkeypatch, {"embedding": {"values": [0.1, 0.2, 0.3]}}, sink=sink)
    vec, tokens = await GeminiEmbedder(api_key="k").embed_with_usage("hello")
    assert vec == [0.1, 0.2, 0.3]
    assert tokens == 0                       # the endpoint reports no usage
    assert ":embedContent" in sink["url"]


@pytest.mark.asyncio
async def test_gemini_forwards_output_dimensionality(monkeypatch):
    sink = {}
    _mock_post(monkeypatch, {"embedding": {"values": [0.0]}}, sink=sink)
    await GeminiEmbedder(api_key="k", dimensions=768).embed("hello")
    assert sink["body"]["outputDimensionality"] == 768


@pytest.mark.asyncio
async def test_gemini_batch_uses_the_batch_endpoint_and_keeps_order(monkeypatch):
    sink = {}
    _mock_post(monkeypatch, {"embeddings": [{"values": [1.0]}, {"values": [2.0]}]}, sink=sink)
    vecs = await GeminiEmbedder(api_key="k").embed_batch(["a", "", "b"])
    assert ":batchEmbedContents" in sink["url"]
    assert len(sink["body"]["requests"]) == 2       # the empty is not sent…
    assert vecs == [[1.0], [], [2.0]]               # …but holds its slot


@pytest.mark.asyncio
async def test_gemini_auth_error_is_typed(monkeypatch):
    _mock_post(monkeypatch, {}, status=403)
    with pytest.raises(InvalidAPIKeyError):
        await GeminiEmbedder(api_key="bad").embed("hello")


# ── Bedrock (boto3, sync in an executor) ──────────────────────────────────────

class _FakeBody:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload)


def _bedrock(model, payload, sink=None):
    e = BedrockEmbedder(model=model, aws_access_key="k", aws_secret_key="s")

    class C:
        def invoke_model(self, **kw):
            if sink is not None:
                sink.update(kw)
                sink["parsed"] = json.loads(kw["body"])
            return {"body": _FakeBody(payload)}

    e._client = C()
    return e


@pytest.mark.asyncio
async def test_bedrock_titan_shape():
    sink = {}
    e = _bedrock("amazon.titan-embed-text-v2:0",
                 {"embedding": [0.5, 0.5], "inputTextTokenCount": 4}, sink)
    vec, tokens = await e.embed_with_usage("hello")
    assert vec == [0.5, 0.5] and tokens == 4
    assert sink["parsed"]["inputText"] == "hello"


@pytest.mark.asyncio
async def test_bedrock_cohere_shape_is_different():
    """Cohere takes a list and answers under a different key — a shared adapter would
    silently return nothing."""
    sink = {}
    e = _bedrock("cohere.embed-multilingual-v3", {"embeddings": [[0.1, 0.2]]}, sink)
    vec, _ = await e.embed_with_usage("hello")
    assert vec == [0.1, 0.2]
    assert sink["parsed"]["texts"] == ["hello"]


@pytest.mark.asyncio
async def test_bedrock_titan_dimensions_forwarded():
    sink = {}
    e = _bedrock("amazon.titan-embed-text-v2:0", {"embedding": [0.0]}, sink)
    e.dimensions = 768
    await e.embed("hello")
    assert sink["parsed"]["dimensions"] == 768


# ── factory routing ────────────────────────────────────────────────────────────

def test_create_embedder_routes_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    e = create_embedder("gemini:text-embedding-004", dimensions=768)
    assert isinstance(e, GeminiEmbedder) and e.dimensions == 768


def test_create_embedder_routes_bedrock(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    e = create_embedder("bedrock:amazon.titan-embed-text-v2:0")
    assert isinstance(e, BedrockEmbedder)
    assert e.model == "amazon.titan-embed-text-v2:0"    # the model id's own colon survives


def test_create_embedder_rejects_a_provider_with_no_embedding_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    with pytest.raises(NotImplementedError):
        create_embedder("groq:whatever")


def test_create_embedder_bare_string_is_ollama():
    assert isinstance(create_embedder("nomic-embed-text"), OllamaEmbedder)


def test_create_embedder_gemini_without_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        create_embedder("gemini:text-embedding-004")
