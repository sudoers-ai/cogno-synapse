"""
cogno_synapse.gemini_embedder — Google Gemini embedding provider.

Talks to the Generative Language REST API (``:embedContent`` /
``:batchEmbedContents``) over ``httpx``, which synapse already depends on.

This deliberately diverges from :class:`GeminiBackend`, which uses the
``google-genai`` SDK: embeddings are a single small POST, so going over REST
keeps Gemini embeddings free of an optional dependency. The endpoint shape is
the one the parent Cogno used in production.

See ``openai_embedder`` for why there is no fallback chain, and why
``dimensions`` is a store constraint rather than a tuning knob. Gemini exposes
it as ``outputDimensionality`` and only newer models honour it.
"""

from __future__ import annotations

import logging
import os

import httpx

from cogno_synapse._math import cosine_similarity
from cogno_synapse.errors import InvalidAPIKeyError

logger = logging.getLogger("cogno_synapse.gemini_embedder")

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiEmbedder:
    """Embedder for Google's Generative Language embedding endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.dimensions = dimensions
        self.timeout = timeout
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set — embedding calls will fail")

    def _request(self, text: str) -> dict:
        body: dict = {"model": f"models/{self.model}",
                      "content": {"parts": [{"text": text}]}}
        if self.dimensions:
            body["outputDimensionality"] = self.dimensions
        return body

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{_BASE}/{self.model}:{path}?key={self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code in (401, 403):
            raise InvalidAPIKeyError(f"Gemini rejected the API key (HTTP {resp.status_code})")
        resp.raise_for_status()
        return resp.json()

    async def embed(self, text: str) -> list[float]:
        vec, _ = await self.embed_with_usage(text)
        return vec

    async def embed_with_usage(self, text: str) -> tuple[list[float], int]:
        """``(vector, tokens)``. The endpoint reports no token usage, so tokens are 0
        — the same convention ``OllamaEmbedder`` uses when the server omits the count.
        A caller that needs a number wraps this in ``CachingEmbedder``."""
        if not text:
            return [], 0
        data = await self._post("embedContent", self._request(text))
        return list(data.get("embedding", {}).get("values", [])), 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs, _ = await self.embed_batch_with_usage(texts)
        return vecs

    async def embed_batch_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """One ``:batchEmbedContents`` call for the whole list — the re-embedding path.
        Empty strings keep their slot without being sent."""
        if not texts:
            return [], 0
        wanted = [(i, t) for i, t in enumerate(texts) if t]
        if not wanted:
            return [[] for _ in texts], 0
        payload = {"requests": [self._request(t) for _, t in wanted]}
        data = await self._post("batchEmbedContents", payload)
        out: list[list[float]] = [[] for _ in texts]
        for (idx, _), item in zip(wanted, data.get("embeddings", [])):
            out[idx] = list(item.get("values", []))
        return out, 0

    async def similarity(self, a: str, b: str) -> float:
        sim, _ = await self.similarity_with_usage(a, b)
        return sim

    async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
        vecs, tokens = await self.embed_batch_with_usage([a, b])
        return cosine_similarity(vecs[0], vecs[1]), tokens
