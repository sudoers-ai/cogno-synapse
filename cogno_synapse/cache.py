from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass

from cogno_synapse.base import Embedder
from cogno_synapse._math import cosine_similarity


@dataclass
class EmbeddingUsage:
    """
    Cumulative embedding accounting collected by a CachingEmbedder.

    - calls:      embed operations that reached the underlying backend
    - cache_hits: embed operations served from the LRU cache (0 tokens)
    - tokens:     tokens reported by the backend for the network calls
    """
    calls: int = 0
    cache_hits: int = 0
    tokens: int = 0

    def add(self, other: "EmbeddingUsage") -> None:
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.tokens += other.tokens


class CachingEmbedder(Embedder):
    """
    Backend-agnostic LRU cache + token accounting around ANY ``Embedder``.

    Wraps any object implementing the ``Embedder`` protocol (Ollama, OpenAI,
    Bedrock, a test stub, ...) and layers on two cross-cutting concerns that are
    independent of the concrete backend:

    - a **bounded LRU cache** keyed by normalized (stripped/lowercased) text,
      which caps memory in long-running processes; and
    - **token/call usage accounting**, so embedding cost is observable the same
      way LLM generate cost is.

    Because these live here rather than inside ``OllamaEmbedder``, every backend
    gets caching for free by composition::

        embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))

    If the wrapped backend exposes ``embed_with_usage`` (returning
    ``(vector, tokens)``), token counts are captured; otherwise tokens are
    reported as 0 and only call/cache counts are tracked.
    """

    def __init__(self, inner: Embedder, cache_size: int = 2048) -> None:
        self._inner = inner
        # Mirror the wrapped model so `.model` and telemetry keep working through
        # the wrapper.
        self.model = getattr(inner, "model", "unknown")
        # Set cache_size=0 to disable caching entirely.
        self._cache_size = cache_size
        self._cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._usage = EmbeddingUsage()

    # ── Usage accounting ────────────────────────────────────────────
    @property
    def usage(self) -> EmbeddingUsage:
        """Cumulative usage since construction (or since the last reset)."""
        return self._usage

    def reset_usage(self) -> None:
        self._usage = EmbeddingUsage()

    # ── Embedder protocol ───────────────────────────────────────────
    async def embed(self, text: str) -> list[float]:
        vec, _ = await self.embed_with_usage(text)
        return vec

    async def embed_with_usage(self, text: str) -> tuple[list[float], int]:
        """Return ``(vector, tokens_used_this_call)``; 0 tokens on a cache hit."""
        if not text:
            return [], 0

        key = text.strip().lower()
        if key in self._cache:
            self._cache.move_to_end(key)  # mark as recently used
            self._usage.cache_hits += 1
            return self._cache[key], 0

        vec, tokens = await self._inner_embed_usage(text)
        self._usage.calls += 1
        self._usage.tokens += tokens

        if vec and self._cache_size > 0:
            self._cache[key] = vec
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)  # evict least-recently-used
        return vec, tokens

    async def similarity(self, a: str, b: str) -> float:
        sim, _ = await self.similarity_with_usage(a, b)
        return sim

    async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
        """Return ``(cosine_similarity, embedding_tokens_used)``."""
        (vec_a, tok_a), (vec_b, tok_b) = await asyncio.gather(
            self.embed_with_usage(a), self.embed_with_usage(b)
        )
        return cosine_similarity(vec_a, vec_b), tok_a + tok_b

    # ── Internals ───────────────────────────────────────────────────
    async def _inner_embed_usage(self, text: str) -> tuple[list[float], int]:
        # Prefer a usage-aware backend; fall back to plain embed (tokens unknown).
        inner_usage = getattr(self._inner, "embed_with_usage", None)
        if inner_usage is not None:
            return await inner_usage(text)
        return await self._inner.embed(text), 0
