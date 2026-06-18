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

from typing import Any, Optional

from cogno_homeo import CircuitBreaker, MetricsSink, RetryPolicy, resilient_call

from cogno_synapse.base import LLMBackend


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

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        async def attempt(b: LLMBackend) -> tuple[str, int, int]:
            result = await b.generate(system, prompt)
            self._last_successful = b
            return result

        return await resilient_call(
            self.backends, attempt,
            breaker=self._breaker, policy=self._policy, metrics=self._metrics,
        )

    async def chat_with_tools(self, messages: Any, tools: Any, tool_choice: Any = None) -> Any:
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
