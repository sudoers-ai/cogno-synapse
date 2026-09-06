"""The BYOK auth probe: the fail-open bias, and the per-provider auth dialect.

Two properties are worth protecting here and they fail in opposite directions:

* the **bias** — only a real rejection (or a blank key) says "invalid"; everything else,
  including our own inability to reach the provider, says "valid";
* the **dialect** — each provider is handed the key the way IT reads keys. A dialect bug is
  the quiet one: the request goes out unauthenticated, the provider answers 401, and a
  perfectly good key is branded invalid for every user of that provider. So the dialect is
  asserted on the request that was actually built, not on the boolean that came back.
"""

from __future__ import annotations

import httpx
import pytest

from cogno_synapse import API_KEY_PROBES, probe_api_key


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status


class _Client:
    """Records every request it is handed, so a test can assert what was SENT."""

    def __init__(self, *, status: int = 200, raise_exc: bool = False) -> None:
        self._status = status
        self._raise = raise_exc
        self.calls: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        self.calls.append((url, dict(headers or {}), dict(params or {})))
        if self._raise:
            raise RuntimeError("network down")
        return _Resp(self._status)


def _patch(monkeypatch, **kw):
    client = _Client(**kw)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: client)
    return client


# ── the bias ──────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_2xx_says_the_key_is_live(monkeypatch):
    _patch(monkeypatch, status=200)
    assert await probe_api_key("openai", "sk-good") is True


@pytest.mark.asyncio
async def test_an_auth_rejection_is_the_only_verdict_that_invalidates(monkeypatch):
    _patch(monkeypatch, status=401)
    assert await probe_api_key("openai", "sk-bad") is False
    _patch(monkeypatch, status=403)
    assert await probe_api_key("anthropic", "bad") is False


@pytest.mark.asyncio
async def test_a_network_error_never_brands_a_legit_key_invalid(monkeypatch):
    """Fail-OPEN: OUR failure to reach the provider is not evidence about THEIR key."""
    _patch(monkeypatch, raise_exc=True)
    assert await probe_api_key("openai", "sk-x") is True


@pytest.mark.asyncio
async def test_an_unprobeable_provider_is_trusted_without_touching_the_network(monkeypatch):
    """A provider the table cannot probe is trusted — but the trust must come from NOT
    calling, not from a call that failed. The fail-open branch turns a raised error into
    ``True`` too, so a raising stub could not tell the two apart; the recorder can."""
    client = _patch(monkeypatch, status=200)
    assert await probe_api_key("mystery-provider", "some-key") is True
    assert client.calls == []
    # …and the recorder really does see calls when the code makes them, so the assertion
    # above is a tripwire and not a vacuous truth.
    assert await probe_api_key("openai", "sk-live") is True
    assert [c[0] for c in client.calls] == ["https://api.openai.com/v1/models"]


@pytest.mark.asyncio
async def test_a_blank_key_is_refused_for_every_provider(monkeypatch):
    """"Unverifiable" is not "empty": the one thing never trusted is a blank key."""
    client = _patch(monkeypatch, status=200)
    for provider in list(API_KEY_PROBES) + ["mystery-provider"]:
        assert await probe_api_key(provider, "") is False
    assert client.calls == []


@pytest.mark.asyncio
async def test_the_provider_name_is_matched_case_insensitively(monkeypatch):
    client = _patch(monkeypatch, status=401)
    assert await probe_api_key("OpenAI", "sk-live") is False   # probed, and rejected
    assert [c[0] for c in client.calls] == ["https://api.openai.com/v1/models"]


# ── the dialect ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,header,param",
    [
        ("openai", ("Authorization", "Bearer K"), None),
        ("groq", ("Authorization", "Bearer K"), None),
        ("xai", ("Authorization", "Bearer K"), None),
        ("anthropic", ("x-api-key", "K"), None),
        ("elevenlabs", ("xi-api-key", "K"), None),
        ("gemini", None, ("key", "K")),
    ],
)
async def test_each_provider_receives_the_key_in_its_own_dialect(
        monkeypatch, provider, header, param):
    client = _patch(monkeypatch, status=200)
    await probe_api_key(provider, "K")
    url, headers, params = client.calls[-1]
    assert url == API_KEY_PROBES[provider][0]
    if header is not None:
        assert headers.get(header[0]) == header[1]
    if param is not None:
        assert params.get(param[0]) == param[1]
    # the key travels in exactly ONE place — a probe that also leaked it into the query
    # string would put a live credential in the provider's access logs.
    carriers = [v for v in headers.values() if "K" in v] + [v for v in params.values() if "K" in v]
    assert len(carriers) == 1


@pytest.mark.asyncio
async def test_anthropic_sends_the_api_version_it_requires(monkeypatch):
    """Anthropic's REST API rejects a request with no ``anthropic-version``, so dropping it
    would turn every valid Anthropic key into an invalid one."""
    client = _patch(monkeypatch, status=200)
    await probe_api_key("anthropic", "K")
    assert client.calls[-1][1].get("anthropic-version") == "2023-06-01"


# ── the caller's own table ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_caller_extends_the_table_instead_of_forking_the_module(monkeypatch):
    client = _patch(monkeypatch, status=200)
    probes = {**API_KEY_PROBES, "myprovider": ("https://example.test/v1/whoami", "bearer")}
    assert await probe_api_key("myprovider", "K", probes=probes) is True
    url, headers, _ = client.calls[-1]
    assert url == "https://example.test/v1/whoami"
    assert headers["Authorization"] == "Bearer K"
    # …and without the override the same provider is unknown, so the parameter is what did it.
    assert await probe_api_key("myprovider", "K") is True
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_an_override_replaces_the_table_rather_than_merging_with_it(monkeypatch):
    """``probes`` is the whole table, not an addendum — a caller that wants the shipped
    providers too spreads ``API_KEY_PROBES`` in. Pinned because a merging implementation
    would silently keep probing providers the caller meant to drop."""
    client = _patch(monkeypatch, status=401)
    assert await probe_api_key("openai", "K", probes={"only": ("https://x.test", "bearer")}) is True
    assert client.calls == []


def test_every_shipped_probe_uses_a_dialect_the_code_implements():
    """A typo in a table entry's auth style is invisible: the request just goes out with no
    credential and the provider answers 401, i.e. a valid key reads invalid."""
    assert {style for _url, style in API_KEY_PROBES.values()} <= {
        "bearer", "anthropic", "gemini", "xi"}
    assert all(url.startswith("https://") for url, _style in API_KEY_PROBES.values())
