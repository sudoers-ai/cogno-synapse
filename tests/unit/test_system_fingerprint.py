"""The provider's identifier for WHO served the call — recorded per call, or honestly absent.

Why this exists, measured: a downstream ALLOW/BLOCK relevance classifier running at
``temperature=0`` through this backend answered ALLOW 4/4 at one time and BLOCK 7/7 forty-five
minutes later, stable within each period, over two process restarts of the same build — on a
prompt proven byte-identical by a digest of the rendered text. The served code was excluded by
digest on both sides. What was left unexamined was the inference backend behind the model
alias, because this layer threw away the one field that names it: a hosted provider only
*requests* greedy decoding at ``temperature=0``, and an alias can move between snapshots and
replicas without the name changing. OpenAI stamps ``system_fingerprint`` on every chat
completion and ``cogno_synapse`` discarded it, so "did the backend change under us?" had no
answer that was not a theory. With it recorded, it is a string comparison.

Two properties carry the whole feature and both are about NOT LYING:

* ``None`` is the honest value for "the provider did not say", never ``""`` and never a
  stand-in. These values exist only to be COMPARED with the next call's, and two blanks
  comparing equal would assert that two calls shared a backend when neither named one;
* a value is per CALL. A stale fingerprint surviving into a call that reported none is a false
  statement about who answered — strictly worse than no statement, because it is believed.

Every "→ None" assertion below is paired with a sibling that produces a REAL value from the
SAME harness. A test of an absence that cannot first demonstrate the presence proves only that
the harness is inert.

No network, no SDK, no keys: the client is replaced with a stub, as everywhere else here.
"""

import pytest

_OMIT = object()


class _Resp:
    """A successful text response.

    ``_OMIT`` means the provider did not include the field AT ALL — the common case for the
    OpenAI-compatible endpoints reached through ``base_url`` — as distinct from including it
    with a null value, which OpenAI itself does. Both must read as ``None`` and neither may
    raise, so they are separate cases here rather than one.
    """

    def __init__(self, *, fingerprint=_OMIT, served=_OMIT, content="hello") -> None:
        self.choices = [type("C", (), {
            "finish_reason": "stop",
            "message": type("M", (), {"content": content, "tool_calls": None})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
        if fingerprint is not _OMIT:
            self.system_fingerprint = fingerprint
        if served is not _OMIT:
            self.model = served


class _ToolResp:
    """A successful tool-calling response — the EGO's path, which must record it too."""

    def __init__(self, *, fingerprint=_OMIT, served=_OMIT) -> None:
        call = type("TC", (), {"id": "c1", "function": type("F", (), {
            "name": "get_weather", "arguments": '{"city":"X"}'})()})()
        self.choices = [type("C", (), {
            "finish_reason": "tool_calls",
            "message": type("M", (), {"content": "", "tool_calls": [call]})()})()]
        self.usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 18})()
        if fingerprint is not _OMIT:
            self.system_fingerprint = fingerprint
        if served is not _OMIT:
            self.model = served


def _client_returning(resp):
    class _Completions:
        async def create(self, **kw):
            if isinstance(resp, Exception):
                raise resp
            return resp

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            return None

    return _Client


def _backend(resp):
    from cogno_synapse.openai_backend import OpenAIBackend

    b = OpenAIBackend(model="gpt-4o-mini", api_key="sk-x")
    b._client = _client_returning(resp)  # type: ignore[method-assign]
    return b


# ── the value is recorded, on BOTH paths ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_records_the_fingerprint_the_provider_stamped():
    from cogno_synapse import system_fingerprint_of

    b = _backend(_Resp(fingerprint="fp_abc"))
    text, tin, tout = await b.generate("sys", "hi")
    assert (text, tin, tout) == ("hello", 10, 5)
    assert system_fingerprint_of(b) == "fp_abc"


@pytest.mark.asyncio
async def test_chat_with_tools_records_it_too():
    """Recording it only on ``generate`` would leave the EGO's whole loop — every tool step and
    every correction retry — unable to say which backend answered."""
    from cogno_synapse import system_fingerprint_of

    b = _backend(_ToolResp(fingerprint="fp_tools"))
    msg, tin, tout = await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert tin == 11 and len(msg["tool_calls"]) == 1
    assert system_fingerprint_of(b) == "fp_tools"


# ── silence reads as None, and the harness proves it can say otherwise ────────────────────

@pytest.mark.asyncio
async def test_a_provider_that_omits_the_field_answers_None_and_does_not_raise():
    """The common case for the OpenAI-COMPATIBLE endpoints (DeepSeek, Moonshot, xAI,
    OpenRouter, Together, Fireworks): no such field on the response at all."""
    from cogno_synapse import system_fingerprint_of

    control = _backend(_Resp(fingerprint="fp_present"))
    await control.generate("sys", "hi")
    assert system_fingerprint_of(control) == "fp_present", "harness cannot produce a presence"

    b = _backend(_Resp())                      # attribute absent entirely
    assert await b.generate("sys", "hi") == ("hello", 10, 5)
    assert system_fingerprint_of(b) is None


@pytest.mark.asyncio
async def test_a_field_present_but_null_answers_None():
    """OpenAI itself returns ``system_fingerprint: null`` for some models. Present-and-null and
    absent are different wire shapes and must reach the same honest answer."""
    from cogno_synapse import system_fingerprint_of

    control = _backend(_Resp(fingerprint="fp_present"))
    await control.generate("sys", "hi")
    assert system_fingerprint_of(control) == "fp_present", "harness cannot produce a presence"

    b = _backend(_Resp(fingerprint=None))
    await b.generate("sys", "hi")
    assert system_fingerprint_of(b) is None


@pytest.mark.asyncio
async def test_a_blank_fingerprint_is_not_a_fingerprint():
    """``""`` would COMPARE EQUAL to the next blank and assert that two calls were served by the
    same configuration when neither named one. Blank is silence, and silence is ``None``."""
    from cogno_synapse import system_fingerprint_of

    control = _backend(_Resp(fingerprint=" fp_padded "))
    await control.generate("sys", "hi")
    assert system_fingerprint_of(control) == "fp_padded", "harness cannot produce a presence"

    for blank in ("", "   ", "\n"):
        b = _backend(_Resp(fingerprint=blank))
        await b.generate("sys", "hi")
        assert system_fingerprint_of(b) is None, f"blank {blank!r} read as a value"


def test_before_any_call_there_is_no_fingerprint():
    from cogno_synapse import system_fingerprint_of
    from cogno_synapse.openai_backend import OpenAIBackend

    assert system_fingerprint_of(OpenAIBackend(model="gpt-4o-mini", api_key="sk-x")) is None


# ── per CALL: the stale value is the failure this must not have ───────────────────────────

@pytest.mark.asyncio
async def test_the_second_call_wins():
    from cogno_synapse import system_fingerprint_of

    b = _backend(_Resp(fingerprint="fp_one"))
    await b.generate("sys", "hi")
    assert system_fingerprint_of(b) == "fp_one"

    b._client = _client_returning(_Resp(fingerprint="fp_two"))  # type: ignore[method-assign]
    await b.generate("sys", "hi")
    assert system_fingerprint_of(b) == "fp_two"


@pytest.mark.asyncio
async def test_a_call_reporting_nothing_does_not_inherit_the_previous_calls_fingerprint():
    """THE test. A stale fingerprint is not a stale number — it is a claim that a named
    backend served a call it did not serve, and it is believed precisely because it looks like
    an answer. The whole feature is telling two calls apart; inheriting makes them look alike."""
    from cogno_synapse import system_fingerprint_of

    b = _backend(_Resp(fingerprint="fp_one"))
    await b.generate("sys", "hi")
    assert system_fingerprint_of(b) == "fp_one"

    b._client = _client_returning(_Resp())  # type: ignore[method-assign]
    await b.generate("sys", "hi")
    assert system_fingerprint_of(b) is None


@pytest.mark.asyncio
async def test_the_tool_path_does_not_inherit_it_either():
    from cogno_synapse import system_fingerprint_of

    b = _backend(_ToolResp(fingerprint="fp_one"))
    await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert system_fingerprint_of(b) == "fp_one"

    b._client = _client_returning(_ToolResp())  # type: ignore[method-assign]
    await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert system_fingerprint_of(b) is None


@pytest.mark.asyncio
async def test_a_FAILED_call_leaves_no_fingerprint_behind():
    """A raise must clear it too: the next reader would otherwise attribute a call that never
    happened to the backend that served the one before it."""
    from cogno_synapse import system_fingerprint_of

    b = _backend(_Resp(fingerprint="fp_one"))
    await b.generate("sys", "hi")
    assert system_fingerprint_of(b) == "fp_one"

    b._client = _client_returning(RuntimeError("boom"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await b.generate("sys", "hi")
    assert system_fingerprint_of(b) is None


# ── a chain answers for the link that actually ran ────────────────────────────────────────

class _Dead:
    """A backend that fails now but carries a fingerprint from some earlier, successful call."""

    model = "dead"

    def __init__(self, fingerprint="fp_stale_from_a_previous_call") -> None:
        self.last_system_fingerprint = fingerprint
        self.last_served_model = "dead-2024-01-01"

    async def generate(self, system, prompt):
        raise RuntimeError("down")


class _Live:
    model = "gpt-4o-mini"

    def __init__(self, fingerprint="fp_live") -> None:
        self._fingerprint = fingerprint
        self.last_system_fingerprint = None
        self.last_served_model = None

    async def generate(self, system, prompt):
        self.last_system_fingerprint = self._fingerprint
        self.last_served_model = "gpt-4o-mini-2024-07-18" if self._fingerprint else None
        return "ok", 10, 5


@pytest.mark.asyncio
async def test_a_fallback_chain_answers_for_the_link_that_actually_ran():
    """``FallbackBackend`` already forwards ``model`` from the successful backend. Reporting the
    FIRST backend's fingerprint for a call the SECOND one served would pair one backend's name
    with another backend's identity — the exact confusion the field exists to remove."""
    from cogno_synapse import FallbackBackend, served_model_of, system_fingerprint_of

    chain = FallbackBackend([_Dead(), _Live()])
    assert system_fingerprint_of(chain) is None          # nothing has run yet

    assert await chain.generate("s", "p") == ("ok", 10, 5)
    assert chain.model == "gpt-4o-mini"
    assert system_fingerprint_of(chain) == "fp_live"
    assert served_model_of(chain) == "gpt-4o-mini-2024-07-18"


@pytest.mark.asyncio
async def test_a_chain_whose_serving_link_reports_nothing_answers_None():
    """Not the dead link's stale value — a fallback to a provider that does not stamp the field
    is exactly when the temptation to fill the gap arises, and filling it would name a backend
    that did not run."""
    from cogno_synapse import FallbackBackend, served_model_of, system_fingerprint_of

    control = FallbackBackend([_Dead(), _Live()])
    await control.generate("s", "p")
    assert system_fingerprint_of(control) == "fp_live", "harness cannot produce a presence"

    chain = FallbackBackend([_Dead(), _Live(fingerprint=None)])
    assert await chain.generate("s", "p") == ("ok", 10, 5)
    assert system_fingerprint_of(chain) is None
    assert served_model_of(chain) is None


# ── a backend with no such notion says so, and does not explode ───────────────────────────

def test_a_backend_that_reports_nothing_answers_None_not_an_error():
    """Anthropic, Gemini, Bedrock, Ollama, a stub, the distilled student: there is no such
    field, and inventing a substitute (a model name, a hash of the request) would compare equal
    across genuinely different backends — the one failure this must not have."""
    from cogno_synapse import served_model_of, system_fingerprint_of

    assert system_fingerprint_of(object()) is None
    assert served_model_of(object()) is None
    assert system_fingerprint_of(type("B", (), {"last_system_fingerprint": None})()) is None
    assert system_fingerprint_of(type("B", (), {"last_system_fingerprint": 7})()) is None
    assert system_fingerprint_of(type("B", (), {"last_system_fingerprint": "fp_x"})()) == "fp_x"


@pytest.mark.asyncio
async def test_the_ollama_backend_answers_None_after_a_real_generate(monkeypatch):
    """Not a constructor check: the REAL ``generate`` runs (with the HTTP client stubbed, no
    server) and still reports nothing, because the local API has no such field — and the reader
    returns ``None`` rather than raising ``AttributeError`` on the way past."""
    from cogno_synapse import ollama as ollama_module
    from cogno_synapse import served_model_of, system_fingerprint_of
    from cogno_synapse.ollama import OllamaBackend

    class _OllamaResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok", "prompt_eval_count": 3, "eval_count": 2}

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            return _OllamaResp()

    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", _FakeClient)

    b = OllamaBackend(model="a-local-model")
    assert system_fingerprint_of(b) is None
    assert await b.generate("s", "p") == ("ok", 3, 2)
    assert system_fingerprint_of(b) is None
    assert served_model_of(b) is None


# ── the other half of the question: WHICH SNAPSHOT answered ───────────────────────────────

@pytest.mark.asyncio
async def test_the_served_model_is_what_answered_not_what_was_asked_for():
    """``backend.model`` is the alias we SENT; ``served_model_of`` is the dated snapshot that
    came back. An alias silently re-pointed at a new snapshot is the other way the backend
    changes under a stable name, and it is a different fact from the fingerprint."""
    from cogno_synapse import served_model_of

    b = _backend(_Resp(fingerprint="fp_abc", served="gpt-4o-mini-2024-07-18"))
    await b.generate("sys", "hi")
    assert b.model == "gpt-4o-mini"
    assert served_model_of(b) == "gpt-4o-mini-2024-07-18"


@pytest.mark.asyncio
async def test_the_served_model_travels_on_the_tool_path_and_does_not_go_stale():
    from cogno_synapse import served_model_of

    b = _backend(_ToolResp(served="gpt-4o-mini-2024-07-18"))
    await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert served_model_of(b) == "gpt-4o-mini-2024-07-18"

    b._client = _client_returning(_ToolResp())  # type: ignore[method-assign]
    await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert served_model_of(b) is None


# ── Groq: OpenAI-shaped response, same field, read the same way ───────────────────────────

@pytest.mark.asyncio
async def test_groq_records_it_too_and_does_not_go_stale():
    from cogno_synapse import served_model_of, system_fingerprint_of
    from cogno_synapse.groq_backend import GroqBackend

    b = GroqBackend(model="llama-3.1-8b-instant", api_key="gsk-x")
    assert system_fingerprint_of(b) is None
    b._client = _client_returning(  # type: ignore[method-assign]
        _Resp(fingerprint="fp_groq", served="llama-3.1-8b-instant"))
    assert await b.generate("sys", "hi") == ("hello", 10, 5)
    assert system_fingerprint_of(b) == "fp_groq"
    assert served_model_of(b) == "llama-3.1-8b-instant"

    b._client = _client_returning(_Resp())  # type: ignore[method-assign]
    await b.generate("sys", "hi")
    assert system_fingerprint_of(b) is None


@pytest.mark.asyncio
async def test_groq_tool_path_records_it_too():
    from cogno_synapse import system_fingerprint_of
    from cogno_synapse.groq_backend import GroqBackend

    b = GroqBackend(model="llama-3.1-8b-instant", api_key="gsk-x")
    b._client = _client_returning(_ToolResp(fingerprint="fp_groq_tools"))  # type: ignore[method-assign]
    msg, tin, _ = await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert tin == 11 and len(msg["tool_calls"]) == 1
    assert system_fingerprint_of(b) == "fp_groq_tools"


def test_both_readers_are_exported_from_the_package_root():
    """Consumers import from ``cogno_synapse``, like ``cached_tokens_of`` beside them — a helper
    reachable only by its module path is a helper every caller re-derives instead."""
    import cogno_synapse

    assert "system_fingerprint_of" in cogno_synapse.__all__
    assert "served_model_of" in cogno_synapse.__all__
    assert cogno_synapse.system_fingerprint_of is not None
    assert cogno_synapse.served_model_of is not None
