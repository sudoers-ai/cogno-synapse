"""The per-call timeout of a CLOUD backend — one number, written once, and why.

Until 0.1.4 the factory gave every backend 600 s, and the SDK clients retried twice on their
own UNDER the host's retry policy (cogno-homeo). A provider that hung therefore held a turn for
up to 600 s per attempt, times 3 SDK attempts, times 3 homeo attempts. A turn measured 1009 s.

Now:

* **30 s per call** (``DEFAULT_CLOUD_TIMEOUT_S``). A healthy chat completion of these
  pipelines answers in a few seconds. A call still silent at 30 s is treated as lost, and the ONE
  retry layer above (the host's ``RetryPolicy`` over ``FallbackBackend``, with its breaker)
  decides what happens next. The cost, declared: a reply that legitimately needs more than
  30 s of generation is cut. That is why the value is steerable.
* **Steerable**: an explicit ``timeout=`` wins; otherwise ``COGNO_CLOUD_TIMEOUT_S``, read per
  construction (not at import), so a host or a test can set it after this module is loaded.
* **The SDK clients never retry** (``max_retries=0``): the retry layer is cogno-homeo's,
  which also holds the circuit breaker. Two stacked layers multiplied the attempts.

Ollama is NOT governed by this: a local model on CPU can take minutes for one generation, so it
keeps its own timeout (``ollama.default_timeout``, ``COGNO_OLLAMA_TIMEOUT``; the factory keeps
passing 600 s, as before).
"""

from __future__ import annotations

import os

#: Seconds a cloud call may take before it is given up (connect, write and the wait for the
#: answer alike).
DEFAULT_CLOUD_TIMEOUT_S = 30.0

#: The SDK's own retries: none. The one retry layer is the host's (cogno-homeo).
SDK_MAX_RETRIES = 0


def cloud_timeout() -> float:
    """``COGNO_CLOUD_TIMEOUT_S`` when it is a positive number, else :data:`DEFAULT_CLOUD_TIMEOUT_S`."""
    raw = os.environ.get("COGNO_CLOUD_TIMEOUT_S", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CLOUD_TIMEOUT_S
    return value if value > 0 else DEFAULT_CLOUD_TIMEOUT_S
