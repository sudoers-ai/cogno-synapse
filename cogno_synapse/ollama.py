from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

from cogno_synapse.base import LLMBackend, Embedder
from cogno_synapse._math import cosine_similarity
from cogno_synapse._obs import log_done, log_request, warn_if_retryable

logger = logging.getLogger("cogno_synapse.ollama")


class OllamaBackend(LLMBackend):
    """
    Concrete LLM backend that calls a local Ollama instance.
    """
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = 8192,
        max_tokens: Optional[int] = 4096,
        format: Optional[str] = None,
        think: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        # Structured decoding: set to "json" so Ollama constrains output to valid
        # JSON (or a JSON schema). The NOUMENO/NER stages consume JSON, so a
        # JSON-producing backend sharply reduces parse failures.
        self.format = format
        # Disable model "thinking" by default. Reasoning models (qwen3, deepseek,
        # …) otherwise route their output to a separate `thinking` field and leave
        # `response` EMPTY → the stages get "" and raise StageParseError. The
        # cognitive stages want direct JSON, not chain-of-thought, so think=False
        # is the right default; it is a harmless no-op on non-reasoning models.
        self.think = think
        self._endpoint = f"{self.base_url}/api/generate"

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        payload: dict = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
        }
        if self.format:
            payload["format"] = self.format
        payload["think"] = self.think
        options: dict = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if options:
            payload["options"] = options

        log_request(logger, "ollama", self.model, system, prompt)
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self._endpoint, json=payload)

        try:
            response.raise_for_status()
        except Exception as exc:
            warn_if_retryable(logger, "ollama", self.model, exc)
            raise
        data = response.json()

        # Prefer `response`; fall back to `thinking` so a reasoning model that
        # (despite think=False) still emitted only to the thinking channel is
        # salvaged instead of yielding an empty string.
        text = data.get("response") or data.get("thinking") or ""
        tokens_in = data.get("prompt_eval_count", 0)
        tokens_out = data.get("eval_count", 0)

        log_done(logger, "ollama", self.model, t0, tokens_in, tokens_out)
        return text, tokens_in, tokens_out

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                await client.get(f"{self.base_url}/api/tags")
            return True
        except Exception:
            return False


class OllamaEmbedder(Embedder):
    """
    Local embedding provider using Ollama's /api/embeddings endpoint.

    This is a thin, stateless client. Caching is intentionally NOT done here —
    wrap it in ``CachingEmbedder`` (cogno_synapse.cache) to add a bounded LRU
    cache and token accounting, so those concerns work for any backend, not
    just Ollama::

        embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))
    """
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed(self, text: str) -> list[float]:
        vec, _ = await self.embed_with_usage(text)
        return vec

    async def embed_with_usage(self, text: str) -> tuple[list[float], int]:
        """Embed ``text`` and report ``(vector, prompt_tokens)``.

        Ollama returns ``prompt_eval_count`` for embedding requests on recent
        versions; older builds omit it, in which case tokens default to 0.
        """
        if not text:
            return [], 0

        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding", [])
        tokens = int(data.get("prompt_eval_count", 0) or 0)
        return embedding, tokens

    async def similarity(self, a: str, b: str) -> float:
        sim, _ = await self.similarity_with_usage(a, b)
        return sim

    async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
        (vec_a, tok_a), (vec_b, tok_b) = await asyncio.gather(
            self.embed_with_usage(a), self.embed_with_usage(b)
        )
        return cosine_similarity(vec_a, vec_b), tok_a + tok_b
