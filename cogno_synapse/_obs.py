"""
cogno_synapse._obs — tiny logging helpers shared by the backends.

Keeps the ``key=value`` log lines byte-identical across the six backends without
a logging framework. Per LOGGING.md: libs emit, the host configures. INFO marks
the (relatively expensive) "call finished" milestone; WARNING flags a recoverable
transport error (429/5xx) just before it propagates to a ``FallbackBackend``;
the request body goes to DEBUG only (dev-only — it carries user content/PII), and
the ``api_key`` is never logged.
"""

from __future__ import annotations

import logging
import time
from typing import Any

# HTTP statuses worth a WARNING (rate limit / transient upstream) before re-raise.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def status_of(exc: Exception) -> Any:
    """Best-effort HTTP status from an SDK/httpx exception (None if absent)."""
    direct = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if direct is not None:
        return direct
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def is_retryable(exc: Exception) -> bool:
    return status_of(exc) in _RETRYABLE_STATUS


def log_request(logger: logging.Logger, provider: str, model: str, system: str, prompt: str) -> None:
    """DEV-ONLY full request trace (DEBUG). Carries user content — never on in prod."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "event=generate provider=%s model=%s system_len=%d prompt_len=%d prompt=%s",
            provider, model, len(system), len(prompt), prompt,
        )


def log_done(
    logger: logging.Logger, provider: str, model: str, t0: float, tokens_in: int, tokens_out: int
) -> None:
    logger.info(
        "event=generate_done provider=%s model=%s latency_ms=%.1f tokens_in=%d tokens_out=%d",
        provider, model, (time.perf_counter() - t0) * 1000.0, tokens_in, tokens_out,
    )


def warn_if_retryable(logger: logging.Logger, provider: str, model: str, exc: Exception) -> None:
    """WARNING on a recoverable transport error before it propagates (fallback catches)."""
    if is_retryable(exc):
        logger.warning(
            "event=generate_retryable provider=%s model=%s status=%s error=%s",
            provider, model, status_of(exc), exc,
        )
