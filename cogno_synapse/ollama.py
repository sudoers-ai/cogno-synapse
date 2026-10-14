from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Mapping, Optional, Union

import httpx

from cogno_synapse.base import LLMBackend, Embedder
from cogno_synapse.schema import (SCHEMA_REFUSED, SchemaRefusals, dialect_for, is_invalid_request,
                                  schema_digest, to_dialect)
from cogno_synapse._math import cosine_similarity
from cogno_synapse._obs import log_done, log_request, warn_if_retryable

logger = logging.getLogger("cogno_synapse.ollama")



# How long to wait on a single Ollama call. 120 s is right for a GPU box; a CPU-only host
# (a small VM, a CI runner) can take longer than that for one generation on an 8B model and
# was dying on httpx.ReadTimeout with nothing wrong but the clock — that is what turned the
# nightly canaries in cogno-anima and cogno-soma red. Env-steerable so the deployment that
# knows its own hardware can say so, without every call site growing a parameter.
_TIMEOUT_ENV = "COGNO_OLLAMA_TIMEOUT"
_DEFAULT_TIMEOUT = 120


def default_timeout() -> int:
    """Seconds to allow one Ollama call: ``$COGNO_OLLAMA_TIMEOUT`` or 120.

    Read per construction, not at import, so a test or a host can set the variable after
    this module is already loaded. A non-numeric or non-positive value falls back to the
    default rather than raising — a bad env var must not take the process down.
    """
    raw = os.environ.get(_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        value = int(raw)
    except ValueError:
        logger.warning("stage=synapse event=bad_timeout_env var=%s value=%r", _TIMEOUT_ENV, raw)
        return _DEFAULT_TIMEOUT
    return value if value > 0 else _DEFAULT_TIMEOUT


class OllamaBackend(LLMBackend):
    """
    Concrete LLM backend that calls a local Ollama instance.
    """
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = 8192,
        max_tokens: Optional[int] = 4096,
        format: Optional[str] = None,
        think: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout if timeout is not None else default_timeout()
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
        # The RECEIPT of the last call's schema — a dialect (it travelled), "refused" (asked for,
        # did not travel) or None (no schema). Written at the END of every call. Read it through
        # ``cogno_synapse.schema_applied_of``, immediately after the await.
        self.last_schema_dialect: Optional[str] = None
        # What THIS instance remembers about schemas the server refused (fifteen minutes).
        self._schema_refusals = SchemaRefusals()

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        return await self._post(system, prompt, self.format)

    def schema_dialect(self, form: str) -> Optional[str]:
        """``ollama_format`` for the ``open`` form; ``None`` for ``strict``.

        Ollama's ``format=<schema>`` already is constrained decoding, and it is the only way this
        server takes a schema. ``strict`` names another provider's closed form: answering it here
        too would make two requested forms one identical request.
        """
        return dialect_for("ollama", form)

    async def generate_structured(
        self, system: str, prompt: str, *, schema: Mapping[str, Any], name: str, form: str,
    ) -> tuple[str, int, int]:
        """``generate`` with the canonical ``schema`` in THIS CALL's ``format`` field.

        ``self.format`` is never touched: one instance serves several stages and several turns at
        once, and a schema written onto it would be read by whichever call ran next. ``name`` is
        not sent (this server has no use for one). Receipt and failure rules are the ones
        ``StructuredBackend`` states: no dialect → plain ``generate``, receipt ``None``; a schema
        that cannot be converted, or a 400, → the call goes without it (``self.format`` as it
        is), receipt ``"refused"``; a 429, a timeout or a 5xx propagates.
        """
        dialect = self.schema_dialect(form)
        if dialect is None:
            return await self.generate(system, prompt)
        try:
            converted = to_dialect(schema, dialect)
        except ValueError as exc:
            logger.warning("event=schema_unconvertible provider=ollama model=%s dialect=%s reason=%s",
                           self.model, dialect, exc)
            return await self._without_the_schema(system, prompt)
        sha = schema_digest(converted)
        if self._schema_refusals.refused(self.model, sha):
            return await self._without_the_schema(system, prompt)
        try:
            result = await self._post(system, prompt, converted)
        except Exception as exc:
            if not is_invalid_request(exc):
                raise
            logger.warning("event=schema_refused provider=ollama model=%s dialect=%s schema_sha=%s",
                           self.model, dialect, sha)
            result = await self._without_the_schema(system, prompt)
            self._schema_refusals.remember(self.model, sha)
            return result
        self.last_schema_dialect = dialect
        return result

    async def _without_the_schema(self, system: str, prompt: str) -> tuple[str, int, int]:
        """Today's ``generate``, with the receipt of a schema that was asked for and did not go."""
        result = await self.generate(system, prompt)
        self.last_schema_dialect = SCHEMA_REFUSED       # no await between the line above and this
        return result

    async def _post(
        self, system: str, prompt: str, fmt: "Union[str, Mapping[str, Any], None]",
    ) -> tuple[str, int, int]:
        """One ``/api/generate`` call. ``fmt`` is the ``format`` of THIS request: ``self.format``
        from ``generate`` (the request it always sent), a schema from ``generate_structured``."""
        payload: dict = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
        }
        if fmt:
            payload["format"] = fmt
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
        # Written after the call's LAST suspension point (leaving the ``async with`` above awaits
        # the client's close): nothing awaits between this line and the caller's next statement,
        # so a structured call finishing on this instance in between cannot leave its receipt
        # for a plain call to be read with. ``generate_structured`` overwrites it on return.
        self.last_schema_dialect = None
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
    Local embedding provider using Ollama's **/api/embed** endpoint.

    This is a thin, stateless client. Caching is intentionally NOT done here —
    wrap it in ``CachingEmbedder`` (cogno_synapse.cache) to add a bounded LRU
    cache and token accounting, so those concerns work for any backend, not
    just Ollama::

        embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))

    **Which endpoint, and why it matters to the bill.** It used to post to
    ``/api/embeddings`` — the LEGACY endpoint, which returns the vector and
    NOTHING ELSE. Measured against a live Ollama 0.20.0 on 2026-09-06:

        POST /api/embeddings  ->  {"embedding": [768 floats]}
        POST /api/embed       ->  {"model", "embeddings": [[768 floats]],
                                   "total_duration", "load_duration",
                                   "prompt_eval_count": 17}

    So ``embed_with_usage`` could only ever report 0, and it did: ``embedding_tokens``
    was 0 in 134 of 134 production traces, and the token ledger held ZERO rows for
    the two stages that embed (``noumeno``, ``id``) across 2 370 embedding rows of
    the whole history. Four fields — ``StageMetrics.embedding_tokens`` /
    ``embedding_calls``, ``PipelineContext.total_embedding_tokens``,
    ``trace.totals.embedding_tokens`` — had no writer, because the writer was asking
    the endpoint that does not answer.

    **The response SHAPE differs and that is the trap**: the legacy one returns
    ``embedding`` (SINGULAR, one vector); the new one returns ``embeddings``
    (PLURAL, a LIST of vectors, because it accepts a list of inputs).

    **An old server falls back instead of failing.** ``/api/embed`` arrived in
    Ollama 0.2; a deployment older than that answers 404. Rather than pin a version
    this library cannot verify at import time, the client MEASURES: on a 404/405 it
    retries the legacy endpoint, keeps the vector, reports 0 tokens (which is exactly
    the old behaviour), and remembers the answer so the round trip is paid ONCE per
    instance. A missing count is a cost that reads low; a raised exception is a dead
    turn — and an embedder must never be the reason a turn dies.
    """
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: Optional[int] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout if timeout is not None else default_timeout()
        # Learned once per instance: an Ollama too old for ``/api/embed`` answers 404 to EVERY
        # call, and this client is used several times per turn. Rediscovering it each time buys
        # nothing and costs a failed round trip per embedding. Same shape as the OpenAI
        # backend's ``_tools_need_effort_none``.
        self._legacy_only = False

    async def embed(self, text: str) -> list[float]:
        vec, _ = await self.embed_with_usage(text)
        return vec

    async def embed_with_usage(self, text: str) -> tuple[list[float], int]:
        """Embed ``text`` and report ``(vector, prompt_tokens)``.

        ``/api/embed`` reports ``prompt_eval_count``; the legacy ``/api/embeddings`` reports
        nothing at all, so a server too old for the former yields ``(vector, 0)`` — the old
        behaviour, kept as a floor rather than an error. See the class docstring.
        """
        if not text:
            return [], 0

        if not self._legacy_only:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/embed",
                                         json={"model": self.model, "input": text})
            # 404/405 = this server does not have the endpoint. Anything else is a real error
            # (a bad model name, an overloaded box) and must propagate, exactly as before —
            # falling back on those would hide a broken deployment behind a silent 0.
            if resp.status_code not in (404, 405):
                resp.raise_for_status()
                data = resp.json()
                # PLURAL, and a LIST OF VECTORS: ``/api/embed`` takes a list of inputs, so one
                # string still comes back wrapped. The legacy endpoint's key is ``embedding``,
                # singular, one vector — reading the new response with the old key silently
                # yields [] and every similarity becomes 0.0.
                vectors = data.get("embeddings") or []
                vector = list(vectors[0]) if vectors else []
                return vector, int(data.get("prompt_eval_count", 0) or 0)
            self._legacy_only = True
            logger.warning(
                "stage=synapse event=embed_endpoint_missing base_url=%s — this Ollama has no "
                "/api/embed (it arrived in 0.2); falling back to the legacy /api/embeddings, "
                "which reports NO token count, so embedding usage will meter as 0 for this "
                "deployment. Upgrade Ollama to bill embeddings.", self.base_url)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/embeddings",
                                     json={"model": self.model, "prompt": text})
        resp.raise_for_status()
        data = resp.json()
        # SINGULAR here — the legacy shape. And no count: measured against a live Ollama
        # 0.20.0, this endpoint's whole response is ``{"embedding": [...]}``.
        return list(data.get("embedding") or []), 0

    async def similarity(self, a: str, b: str) -> float:
        sim, _ = await self.similarity_with_usage(a, b)
        return sim

    async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
        (vec_a, tok_a), (vec_b, tok_b) = await asyncio.gather(
            self.embed_with_usage(a), self.embed_with_usage(b)
        )
        return cosine_similarity(vec_a, vec_b), tok_a + tok_b
