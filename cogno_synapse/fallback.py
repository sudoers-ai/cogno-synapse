"""
cogno_synapse.fallback — try a chain of backends, in order, until one succeeds.

A composable resilience primitive: wrap an ordered list of backends; each call
tries them in turn, returning the first success and re-raising the last error if
all fail. Backends that lack native FC are skipped for ``chat_with_tools``.

The failover loop runs over :func:`cogno_homeo.resilient_call`, so a host can opt
into a circuit breaker, retry/backoff and a metrics seam by passing them in —
with none supplied it degrades to the historical "try each once, fail over"
behaviour. Pairs with the adapted backends that **raise** on failure (so this can
catch and fail over).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from cogno_homeo import CircuitBreaker, MetricsSink, RetryPolicy, resilient_call

from cogno_synapse.base import LLMBackend, schema_applied_of, schema_dialect_of
from cogno_synapse.schema import CHAIN_DIALECT


def _supports(backend: object) -> bool:
    return bool(getattr(backend, "supports_native_tools", lambda: False)())


class FallbackBackend:
    """Ordered failover across backends; first success wins, last error propagates."""

    def __init__(
        self,
        backends: list[LLMBackend],
        *,
        breaker: Optional[CircuitBreaker] = None,
        policy: Optional[RetryPolicy] = None,
        metrics: Optional[MetricsSink] = None,
    ) -> None:
        if not backends:
            raise ValueError("FallbackBackend requires at least one backend")
        self.backends = backends
        self._breaker = breaker
        self._policy = policy
        self._metrics = metrics
        self._last_successful: LLMBackend | None = None
        # The RECEIPT of the last call's schema, as the link that SERVED it left it — captured
        # inside the attempt, on the line after the link's own await, and stored here. Not a
        # property forwarded later from ``_last_successful``: a link is shared too, and by the
        # time a later read reached it another call could have overwritten its value.
        self.last_schema_dialect: Optional[str] = None

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        self.last_schema_dialect = None

        async def attempt(b: LLMBackend) -> tuple[str, int, int]:
            result = await b.generate(system, prompt)
            self._last_successful = b
            self.last_schema_dialect = None
            return result

        return await resilient_call(
            self.backends, attempt,
            breaker=self._breaker, policy=self._policy, metrics=self._metrics,
        )

    def schema_dialect(self, form: str) -> Optional[str]:
        """``"chain"`` when AT LEAST ONE link answers this ``form``; ``None`` when none does.

        ANY, not ALL — the opposite of what a chain may promise about native tools, and for a
        reason: the tool announcement picks the PROMPT (native or tagged), so one link that cannot
        keep it breaks the turn. A schema travels OUTSIDE the prompt: the prompt is the same on
        both paths, and a link with no dialect serves the very same call through ``generate``.
        """
        return CHAIN_DIALECT if any(schema_dialect_of(b, form) for b in self.backends) else None

    async def generate_structured(
        self, system: str, prompt: str, *, schema: Mapping[str, Any], name: str, form: str,
    ) -> tuple[str, int, int]:
        """``generate`` over the chain, each link carrying the schema if IT can.

        Decided PER LINK, inside the attempt: a link that answers ``form`` gets
        ``generate_structured`` with the schema EXACTLY as it arrived here; a link that does not
        gets today's two-argument ``generate``. This chain converts nothing — only the leaf knows
        its dialect, and a chain inside a chain is just a link that answers ``"chain"``.

        The receipt is the serving link's, read on the line after its await. Failover is
        untouched: a refused schema is settled INSIDE the leaf (one repeat without it), so no
        retry is spent and no breaker opens; a 429, a timeout or a 5xx raises out of the link and
        moves this loop on, exactly as in ``generate``.
        """
        self.last_schema_dialect = None

        async def attempt(b: LLMBackend) -> tuple[str, int, int]:
            if schema_dialect_of(b, form) is None:
                result = await b.generate(system, prompt)
                receipt = None
            else:
                result = await b.generate_structured(  # type: ignore[attr-defined]
                    system, prompt, schema=schema, name=name, form=form)
                receipt = schema_applied_of(b)      # no await between the line above and this
            self._last_successful = b
            self.last_schema_dialect = receipt
            return result

        return await resilient_call(
            self.backends, attempt,
            breaker=self._breaker, policy=self._policy, metrics=self._metrics,
        )

    async def chat_with_tools(self, messages: Any, tools: Any, tool_choice: Any = None) -> Any:
        self.last_schema_dialect = None          # a tool call carries no schema
        fc_backends = [b for b in self.backends if _supports(b)]
        if not fc_backends:
            raise RuntimeError("FallbackBackend: no backend in the chain supports native tools")

        async def attempt(b: LLMBackend) -> Any:
            result = await b.chat_with_tools(messages, tools, tool_choice)  # type: ignore[attr-defined]
            self._last_successful = b
            return result

        return await resilient_call(
            fc_backends, attempt,
            breaker=self._breaker, policy=self._policy, metrics=self._metrics,
        )

    def supports_native_tools(self) -> bool:
        return any(_supports(b) for b in self.backends)

    @property
    def model(self) -> str:
        active = self._last_successful or self.backends[0]
        return getattr(active, "model", "unknown")

    @property
    def last_cached_tokens(self) -> int:
        """Forwarded from the backend that actually ran, like ``model`` beside it.

        A chain answers for the link that served the call. Without this the ledger would
        attribute the SECOND backend's model to the FIRST one's cache count — or, worse,
        report 0 for a call the provider did discount, because the wrapper has no attribute
        of its own and ``cached_tokens_of`` would fall through to its default.

        Read through ``cogno_synapse.cached_tokens_of``, immediately after the await."""
        active = self._last_successful
        return max(0, int(getattr(active, "last_cached_tokens", 0) or 0)) if active else 0

    @property
    def last_system_fingerprint(self) -> Optional[str]:
        """Forwarded from the backend that actually ran, like ``model`` beside it.

        A chain answers for the link that served the call. Reporting the FIRST backend's
        fingerprint for a call the SECOND one served is not an approximation — it is a false
        statement about who answered, and the whole point of the field is telling one backend
        configuration from another. Before anything has run, and whenever the link that did run
        reports nothing, the answer is ``None``: the chain never fills a gap with a value from a
        backend that did not serve this call.

        Read through ``cogno_synapse.system_fingerprint_of``, immediately after the await."""
        return getattr(self._last_successful, "last_system_fingerprint", None)

    @property
    def last_served_model(self) -> Optional[str]:
        """The snapshot the serving link's provider echoed back. Same rule as above."""
        return getattr(self._last_successful, "last_served_model", None)
