"""Is this provider API key live? — a cost-free auth probe (BYOK key validation).

A deployment that lets its users bring their own provider keys (BYOK) has to answer one
question the moment a key is pasted: *does it work?* The answer drives whatever the caller
gates on it — a dashboard badge, which cloud models unlock, whether the key is stored at all —
so the probe has to be cheap, side-effect free, and fast.

Every provider offers the same cheap shape: an authenticated GET that costs nothing
(list-models, or ``/user``). This module is that one call plus the per-provider auth dialect
(``Authorization: Bearer``, ``x-api-key``, a ``key`` query parameter, ``xi-api-key``).

**The bias is deliberate and it is fail-OPEN.** A clear auth rejection (401/403) → invalid;
a 2xx → valid; anything else — a 5xx, a timeout, DNS down, an unknown provider — → **valid**.
Our failure to reach a provider must never brand a legitimate key invalid: the key is
re-checked for real the first time it is actually used, and a wrongly-invalidated key locks a
paying user out of their own models with no way to tell why. The one thing that is never
trusted is a *blank* key: "unverifiable" is not "empty".

The default table covers the providers a BYOK console usually offers, which is not the same
list as the providers :mod:`cogno_synapse.factory` can build a backend for — a voice provider
is auth-probed exactly like a model provider, and the caller's console lists them side by
side. Splitting the table by which lib "owns" the provider would put one fact in two homes,
which is the drift a single table exists to prevent. A caller with a provider this table does
not know passes its own ``probes`` mapping rather than forking the module; the caller — never
this module — owns which providers it offers.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

import httpx

log = logging.getLogger(__name__)

# provider → (url, auth-style). ``bearer`` covers the OpenAI-compatible providers, whose
# list-models call is free. Public because the caller's own provider list is a SECOND copy of
# this fact: exporting the table lets that caller pin the two together instead of discovering
# the drift as a key that is silently trusted without ever being probed.
API_KEY_PROBES: "dict[str, tuple[str, str]]" = {
    "openai": ("https://api.openai.com/v1/models", "bearer"),
    "groq": ("https://api.groq.com/openai/v1/models", "bearer"),
    "xai": ("https://api.x.ai/v1/models", "bearer"),
    "anthropic": ("https://api.anthropic.com/v1/models", "anthropic"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models", "gemini"),
    "elevenlabs": ("https://api.elevenlabs.io/v1/user", "xi"),
}


def _auth(api_key: str, style: str) -> "tuple[dict[str, str], dict[str, str]]":
    """``(headers, params)`` carrying ``api_key`` in ``style``'s dialect."""
    headers: "dict[str, str]" = {}
    params: "dict[str, str]" = {}
    if style == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    elif style == "anthropic":
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif style == "gemini":
        params["key"] = api_key
    elif style == "xi":
        headers["xi-api-key"] = api_key
    return headers, params


async def probe_api_key(
    provider: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    probes: Optional[Mapping[str, Tuple[str, str]]] = None,
) -> bool:
    """Is ``api_key`` accepted by ``provider``? A cost-free auth probe.

    ``probes`` overrides the shipped :data:`API_KEY_PROBES` table (pass
    ``{**API_KEY_PROBES, "myprovider": (url, "bearer")}`` to extend it).

    Returns ``False`` only for a blank key or a provider that actively rejected it (401/403,
    or any other 4xx/5xx from a provider we did reach). A provider this table cannot probe, or
    a transport error, returns ``True`` for a non-empty key — see the fail-open bias in the
    module docstring."""
    table = API_KEY_PROBES if probes is None else probes
    spec = table.get((provider or "").lower())
    if spec is None or not api_key:
        return bool(api_key)                     # unknown provider → trust a non-empty key
    url, style = spec
    headers, params = _auth(api_key, style)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers, params=params)
    except Exception as exc:  # noqa: BLE001 — a network error must not brand a legit key invalid
        log.warning("event=byok_probe_error provider=%s error=%s — treating as valid", provider, exc)
        return True
    if resp.status_code in (401, 403):
        return False
    return resp.status_code < 400
