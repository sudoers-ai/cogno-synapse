"""The BYOK auth probe: the verdict bias, and the per-provider auth dialect.

Two properties are worth protecting here and they fail in opposite directions:

* the **bias** — only a statement the provider made about THIS CREDENTIAL (401/403), or a
  blank key, says "invalid"; every other outcome is inconclusive and says "valid", a 429, a
  5xx and a 404 included. The blast radius of that rule is not described here but SWEPT, over
  every status 100–599, against the rule it replaced;
* the **dialect** — each provider is handed the key the way IT reads keys. A dialect bug is
  the quiet one: the request goes out unauthenticated, the provider answers 401, and a
  perfectly good key is branded invalid for every user of that provider. So the dialect is
  asserted on the request that was actually built, not on the boolean that came back.
"""

from __future__ import annotations

import logging
import pathlib
import re

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
async def test_an_auth_rejection_is_the_only_status_that_invalidates(monkeypatch):
    """The half of the old rule that SURVIVES. It is also the half the dialect tests lean on:
    a dialect bug sends the request out with no credential, the provider answers 401, and the
    key reads dead — so 401 must keep meaning what it means, or that cover goes with it."""
    _patch(monkeypatch, status=401)
    assert await probe_api_key("openai", "sk-bad") is False
    _patch(monkeypatch, status=403)
    assert await probe_api_key("anthropic", "bad") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 402, 404, 408, 422, 429, 500, 502, 503, 504])
async def test_an_inconclusive_status_never_brands_a_key_dead(monkeypatch, status):
    """A status that is not a statement about THIS CREDENTIAL must not invalidate it.

    A 429 is a rate limit on our IP or their account, a 5xx is their infrastructure, a 404 is
    OUR probe URL — none of them says anything about the key, so none of them may kill it. The
    two errors are not symmetric: a false *invalid* takes a working provider away from a paying
    tenant and, in the one caller that exists, never heals (it probes on save and never again);
    a false *valid* leaves a dead key looking alive until its first real call fails, which the
    caller sees and can act on.

    The two entries that cost the most are named so nobody has to rediscover them: **402** is a
    real key on an unpaid account — authentic, and unusable — and it now reads live; **404** is
    a rotted probe URL, which waves through every key of that provider unprobed. Both are
    accepted, and both are why this branch logs (see the twin below).
    """
    _patch(monkeypatch, status=status)
    assert await probe_api_key("openai", "sk-live-but-throttled") is True


@pytest.mark.asyncio
async def test_the_new_bias_changes_exactly_the_statuses_it_says_it_changes(monkeypatch):
    """The over-tightening probe, swept status by status over the WHOLE corpus (100–599).

    The old rule was ``status_code < 400``; the new one is ``status_code not in (401, 403)``.
    Widening a predicate has victims, so the difference is enumerated here rather than
    described: everything from 400 up EXCEPT the two credential rejections now reads valid
    where it used to read dead, and *nothing else moves* — no 2xx and no 3xx changes hands.
    An edit that widens or narrows that set (exempting a 402, letting a 400 invalidate again)
    fails here by name instead of being left for a reader to notice.
    """
    client = _patch(monkeypatch, status=200)
    changed = set()
    for status in range(100, 600):
        client._status = status
        was_valid = status < 400                       # the rule this PR replaces, frozen
        if (await probe_api_key("openai", "sk-x")) != was_valid:
            changed.add(status)
    assert changed == set(range(400, 600)) - {401, 403}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 429, 500])
async def test_the_waived_statuses_are_logged_so_a_rotted_probe_url_is_not_silent(
        monkeypatch, caplog, status):
    """Fail-open on the VERDICT must not mean fail-silent on the SIGNAL.

    Before this change a rotted probe URL was loud in the worst way — every key of that
    provider read invalid, so somebody filed a bug. Now it reads valid, which is the right
    verdict and the wrong amount of noise: without this line the probe would quietly become a
    no-op for that provider and nobody would ever learn. The verdict fails open; the record
    does not disappear.
    """
    _patch(monkeypatch, status=status)
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.key_probe"):
        assert await probe_api_key("openai", "sk-x") is True
    waivers = [r.getMessage() for r in caplog.records if "byok_probe_inconclusive" in r.getMessage()]
    assert waivers, "the waived status was not recorded anywhere"
    # the STATUS travels too — "something was waived" without saying what is not a signal.
    assert any(f"status={status}" in m for m in waivers)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,why", [(200, "a clean pass has nothing to report"),
                                        (401, "a rejection is a verdict, not a waiver")])
async def test_nothing_but_a_waiver_is_logged_as_inconclusive(monkeypatch, caplog, status, why):
    """The twin of the test above: without it, a log line emitted on EVERY response would
    satisfy that one and say nothing. ``inconclusive`` has to mean the branch that waived."""
    _patch(monkeypatch, status=status)
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.key_probe"):
        await probe_api_key("openai", "sk-x")
    assert not any("byok_probe_inconclusive" in r.getMessage() for r in caplog.records), why


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


# ── the rule is stated three times; nothing used to compare the three ─────────────────

def test_the_prose_quotes_the_verdict_expression_the_code_actually_runs():
    """Three statements of one rule diverged because nothing compared them.

    The verdict lives in exactly one expression. The three long-form statements of it — this
    module's docstring, ``probe_api_key``'s own docstring, and the README's BYOK section —
    are three copies of that one fact, and two of them had drifted to say its OPPOSITE with
    nobody noticing, because prose has no gate. This is that gate, in the mould of the
    prompt/code alphabet pins: it lifts the expression out of the source and demands every
    prose site quote it verbatim, so a change to the verdict turns the prose red instead of
    leaving it to rot.
    """
    from cogno_synapse import key_probe

    src = pathlib.Path(key_probe.__file__).read_text(encoding="utf-8")
    verdict = re.search(r"^\s*return resp\.status_code\s+(.+?)\s*$", src, re.M)
    assert verdict, (
        "the verdict is no longer a single `return resp.status_code <expr>` line — move this "
        "pin to wherever it went, and update every prose site in the same commit"
    )
    quoted = f"status_code {verdict.group(1)}"

    readme = pathlib.Path(__file__).resolve().parents[2] / "README.md"
    assert readme.is_file(), f"README not found at {readme}"
    for name, text in (("module docstring", key_probe.__doc__ or ""),
                       ("function docstring", probe_api_key.__doc__ or ""),
                       ("README", readme.read_text(encoding="utf-8"))):
        assert quoted in text, f"{name} does not quote the verdict the code runs: {quoted!r}"
