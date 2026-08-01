"""
cogno_synapse.bedrock_embedder — AWS Bedrock embedding provider.

Covers the two families the catalog offers: Amazon Titan
(``amazon.titan-embed-text-v*``) and Cohere (``cohere.embed-*``). They do NOT
share a payload shape, so the model id selects the adapter:

* Titan  — ``{"inputText": str}`` → ``{"embedding": [...], "inputTextTokenCount": n}``,
  one text per call, and ``dimensions`` travels as ``dimensions`` (v2 only).
* Cohere — ``{"texts": [str], "input_type": ...}`` → ``{"embeddings": [[...]]}``,
  natively batched, and the output width is fixed by the model.

boto3 is synchronous, so calls go through ``run_in_executor`` — the same pattern
:class:`BedrockBackend` uses.

See ``openai_embedder`` for why there is no fallback chain.

Optional dependency: ``pip install "cogno-synapse[bedrock]"`` (or ``boto3``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from cogno_synapse._math import cosine_similarity

logger = logging.getLogger("cogno_synapse.bedrock_embedder")


class BedrockEmbedder:
    """Embedder for AWS Bedrock's Titan / Cohere embedding models."""

    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
        dimensions: int | None = None,
        timeout: int = 120,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_region: str | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.aws_access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY_ID", "")
        self.aws_secret_key = aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self._client = None
        if not (self.aws_access_key and self.aws_secret_key):
            logger.warning("AWS credentials not fully set — Bedrock embedding calls may fail")

    @property
    def _is_cohere(self) -> bool:
        return self.model.startswith("cohere.")

    def _get_client(self):  # noqa: ANN202 - boto3 types are optional at import time
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError(
                'boto3 not installed. Run: pip install "cogno-synapse[bedrock]"') from exc
        self._client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            config=Config(region_name=self.aws_region, read_timeout=self.timeout,
                          connect_timeout=15, retries={"max_attempts": 0}),
        )
        return self._client

    async def _invoke(self, body: dict) -> dict:
        loop = asyncio.get_running_loop()

        def _call():
            return self._get_client().invoke_model(
                modelId=self.model, body=json.dumps(body),
                accept="application/json", contentType="application/json")
        resp = await loop.run_in_executor(None, _call)
        return json.loads(resp["body"].read())

    async def embed(self, text: str) -> list[float]:
        vec, _ = await self.embed_with_usage(text)
        return vec

    async def embed_with_usage(self, text: str) -> tuple[list[float], int]:
        if not text:
            return [], 0
        if self._is_cohere:
            data = await self._invoke({"texts": [text], "input_type": "search_document"})
            vecs = data.get("embeddings") or [[]]
            return list(vecs[0]), 0
        body: dict = {"inputText": text}
        if self.dimensions:
            body["dimensions"] = self.dimensions          # Titan v2 only; v1 ignores it
        data = await self._invoke(body)
        return list(data.get("embedding", [])), int(data.get("inputTextTokenCount", 0) or 0)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs, _ = await self.embed_batch_with_usage(texts)
        return vecs

    async def embed_batch_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Cohere batches natively; Titan does not, so it falls back to concurrent
        single calls rather than pretending a batch endpoint exists."""
        if not texts:
            return [], 0
        wanted = [(i, t) for i, t in enumerate(texts) if t]
        if not wanted:
            return [[] for _ in texts], 0
        out: list[list[float]] = [[] for _ in texts]

        if self._is_cohere:
            data = await self._invoke({"texts": [t for _, t in wanted],
                                       "input_type": "search_document"})
            for (idx, _), vec in zip(wanted, data.get("embeddings", [])):
                out[idx] = list(vec)
            return out, 0

        results = await asyncio.gather(*(self.embed_with_usage(t) for _, t in wanted))
        tokens = 0
        for (idx, _), (vec, tok) in zip(wanted, results):
            out[idx] = vec
            tokens += tok
        return out, tokens

    async def similarity(self, a: str, b: str) -> float:
        sim, _ = await self.similarity_with_usage(a, b)
        return sim

    async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
        vecs, tokens = await self.embed_batch_with_usage([a, b])
        return cosine_similarity(vecs[0], vecs[1]), tokens
