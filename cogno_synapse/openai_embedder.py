"""
cogno_synapse.openai_embedder — OpenAI-compatible embedding provider.

OpenAI's ``/embeddings`` endpoint (text-embedding-3-small / -large / ada-002).
Implements the ``Embedder`` protocol, so any consumer that depends on the
protocol (cogno-anima's stages, the host's memory/persona routing) takes it
without a change.

Like ``OllamaEmbedder`` this is a thin client — caching and token accounting
belong to ``CachingEmbedder``, so they work for every provider::

    embedder = CachingEmbedder(OpenAIEmbedder(model="text-embedding-3-small"))

**The ``dimensions`` parameter is load-bearing, not a tuning knob.** A vector
store's column is fixed width: pgvector ``vector(768)`` rejects a 1536-float row
outright. The v3 models accept a shortened output, so pass the dimension the
store was built for and a cloud embedder drops into a schema sized for a local
one. ada-002 ignores it (fixed 1536) — the caller must size the column to match.

Deliberately NO fallback chain here (the parent's rule, ported verbatim in
spirit): a vector store mixes embedding spaces the moment a failover swaps
providers mid-write, and cosine over two different spaces returns a plausible
number that means nothing. An embedding call must use exactly what was asked
for, or fail loudly.

Optional dependency: ``pip install "cogno-synapse[openai]"`` (or ``openai``).
"""

from __future__ import annotations

import logging
import os

from cogno_synapse._math import cosine_similarity
from cogno_synapse.errors import InvalidAPIKeyError

logger = logging.getLogger("cogno_synapse.openai_embedder")


def _is_auth_error(exc: Exception) -> bool:
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    return status in (401, 403)


class OpenAIEmbedder:
    """Embedder for OpenAI's ``/embeddings`` API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout: int = 120,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.dimensions = dimensions
        self.timeout = timeout
        self.base_url = base_url
        # One lazily-built AsyncClient reused for the embedder's lifetime. The parent
        # measured 13-16s of repeated TLS handshakes when a client was built per call
        # (worst on skill/persona warm-up, which embeds a burst back-to-back).
        self._client = None
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set — embedding calls will fail")

    def _get_client(self):  # noqa: ANN202 - the SDK type is optional at import time
        if self._client is None:
            import openai
            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.AsyncClient(**kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        vec, _ = await self.embed_with_usage(text)
        return vec

    async def embed_with_usage(self, text: str) -> tuple[list[float], int]:
        """Embed ``text`` and report ``(vector, prompt_tokens)``.

        Tokens come from the API's own ``usage.prompt_tokens`` — not a ``len//4``
        heuristic — so the host bills what the provider actually counted.
        """
        if not text:
            return [], 0
        vecs, tokens = await self.embed_batch_with_usage([text])
        return (vecs[0] if vecs else []), tokens

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs, _ = await self.embed_batch_with_usage(texts)
        return vecs

    async def embed_batch_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Embed many texts in ONE request — the re-embedding path (a model switch
        rewrites every stored vector) is otherwise N sequential round trips.

        Empty strings are preserved as empty vectors in their original position
        rather than sent to the API.
        """
        if not texts:
            return [], 0
        wanted = [(i, t) for i, t in enumerate(texts) if t]
        if not wanted:
            return [[] for _ in texts], 0

        params: dict = {"input": [t for _, t in wanted], "model": self.model}
        if self.dimensions:
            params["dimensions"] = self.dimensions
        try:
            resp = await self._get_client().embeddings.create(**params)
        except Exception as exc:  # noqa: BLE001 — classify, then propagate
            if _is_auth_error(exc):
                raise InvalidAPIKeyError(f"OpenAI rejected the API key: {exc}") from exc
            raise

        # Map by the response's own ``index`` rather than by arrival order. The API documents
        # input order, but pairing a vector with the WRONG text is exactly the silent failure
        # this module exists to avoid — a mis-paired embedding corrupts recall with no error.
        out: list[list[float]] = [[] for _ in texts]
        for item in resp.data:
            pos = getattr(item, "index", None)
            idx = wanted[pos][0] if pos is not None and pos < len(wanted) else None
            if idx is not None:
                out[idx] = list(item.embedding)
        tokens = int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0)
        return out, tokens

    async def similarity(self, a: str, b: str) -> float:
        sim, _ = await self.similarity_with_usage(a, b)
        return sim

    async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
        # One request for the pair — halves the round trips of the hot path
        # (NOUMENO drift + subject continuity, the ID's goal similarity).
        vecs, tokens = await self.embed_batch_with_usage([a, b])
        return cosine_similarity(vecs[0], vecs[1]), tokens
