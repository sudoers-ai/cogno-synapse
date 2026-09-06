"""Is this provider API key live? — a cost-free auth probe (BYOK key validation).

A deployment that lets its users bring their own provider keys (BYOK) has to answer one
question the moment a key is pasted: *does it work?* The answer drives whatever the caller
gates on it — a dashboard badge, which cloud models unlock, whether the key is stored at all —
so the probe has to be cheap, side-effect free, and fast.

Every provider offers the same cheap shape: an authenticated GET that costs nothing
(list-models, or ``/user``). This module is that one call plus the per-provider auth dialect
(``Authorization: Bearer``, ``x-api-key``, a ``key`` query parameter, ``xi-api-key``).

**The bias is fail-OPEN, and the line it draws is "did the provider make a statement about
THIS CREDENTIAL?"** Only a 401 and a 403 do: the verdict is ``status_code not in (401, 403)``.
Everything else is *inconclusive*, and inconclusive reads as **valid** — a timeout, DNS down,
a provider this table cannot probe, and equally a 429, a 500 or a 404 from a provider we did
reach. A 429 is a rate limit on our IP or their account, a 5xx is their infrastructure, a 404
is *our* URL: none of the three is evidence about the key, and a wrongly-invalidated key locks
a paying user out of their own models with no way to tell why.

The asymmetry is deliberate, because the two errors do not cost the same. A false *invalid*
takes a working provider away from a paying tenant; a false *valid* leaves a dead key looking
alive until its first real call fails. Only the first of those is silent to the user who could
fix it, and — measured in the one caller that exists — only the first is **permanent**: that
caller probes on save and never again, so a spurious rejection never heals on its own.

What this bias gives up is stated rather than hidden: a rotted probe URL (404) and an auth
dialect a provider answers with a 400 rather than a 401 now read as valid, i.e. quietly
unprobed. So the inconclusive branch **logs** (``event=byok_probe_inconclusive``) — the
verdict fails open, the signal does not disappear — and the dialect assertions in
``tests/unit/test_key_probe.py``, which check the request that was BUILT rather than the
boolean that came back, become the load-bearing cover for that half.

The one thing that is never trusted is a *blank* key: "unverifiable" is not "empty".

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

    Returns ``False`` for a blank key and for a credential the provider actively rejected —
    and nothing else: the verdict is ``status_code not in (401, 403)``. Every other outcome is
    inconclusive and reads ``True`` for a non-empty key: a transport error, a provider this
    table cannot probe, and any other status, a 429/404/5xx included (those are logged as
    ``event=byok_probe_inconclusive``). See the fail-open bias in the module docstring."""
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
    if resp.status_code >= 400 and resp.status_code not in (401, 403):
        # Inconclusive, not a rejection — but this branch now WAIVES the key, so it must not be
        # silent: a 404 here means the probe URL rotted and every key of this provider is being
        # waved through unprobed, which is exactly the failure a quiet fail-open would hide.
        log.warning("event=byok_probe_inconclusive provider=%s status=%s — treating as valid",
                    provider, resp.status_code)
    return resp.status_code not in (401, 403)
