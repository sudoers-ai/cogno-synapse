"""Typed errors for the model-transport layer.

Backends **raise** on transport/auth failure (they do not silently degrade to
``("", 0, 0)``) so a caller — or ``FallbackBackend`` — can catch, fail over, or
surface the problem distinctly. Auth errors are config problems retry/fallback
cannot fix, so they are typed apart from transient failures.
"""

from __future__ import annotations


class SynapseError(RuntimeError):
    """Base class for all cogno_synapse errors."""


class MissingAPIKeyError(SynapseError):
    """A cloud-provider model was requested but its API key is missing/placeholder.

    Raised by the backend factory when a specific cloud provider is asked for
    without a usable key — fail loudly instead of silently degrading to a weaker
    local model the caller did not ask for.
    """


class InvalidAPIKeyError(SynapseError):
    """A cloud API rejected the provided key at runtime (401/403).

    Unlike transient errors (timeouts, rate limits), an auth error is a config
    problem that retry/fallback cannot fix — backends raise this so the host can
    surface it distinctly rather than treating it as "the model had nothing to say".
    """
