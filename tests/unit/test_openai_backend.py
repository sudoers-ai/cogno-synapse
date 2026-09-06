"""Unit tests for cogno_synapse.openai_backend — no network, no SDK required."""

import pytest

# ── the provider telling us it cut the answer ─────────────────────────────────────────────

class _Choice:
    def __init__(self, finish_reason: str, content: str = "hi") -> None:
        self.finish_reason = finish_reason
        self.message = type("M", (), {"content": content, "tool_calls": None})()


class _Resp:
    def __init__(self, finish_reason: str) -> None:
        self.choices = [_Choice(finish_reason)]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()


@pytest.mark.parametrize("reason, warns", [
    ("length", True),            # hit the token ceiling
    ("content_filter", True),    # cut for policy
    ("stop", False),             # complete
    ("tool_calls", False),       # complete, chose a tool
    ("", False),                 # provider said nothing
])
def test_truncation_is_logged_from_finish_reason(reason, warns, caplog):
    """A cut response is indistinguishable from a complete one here — same shape, same type,
    no exception — so it travels on and fails wherever it happens to break, diagnosed as that
    thing. Measured 2026-08-04: a NOUMENO payload ended mid-string, raised StageParseError and
    killed the turn; the report said "bad JSON", which is what the parser saw and not what
    happened. `finish_reason` is the provider saying so plainly, and nothing read it."""
    import logging

    from cogno_synapse.openai_backend import _warn_if_truncated

    with caplog.at_level(logging.WARNING, logger="cogno_synapse.openai"):
        assert _warn_if_truncated(_Resp(reason), "gpt-4o-mini") is warns
    assert ("truncated_response" in caplog.text) is warns


def test_a_malformed_response_does_not_raise_from_the_detector():
    """Fail-soft: the detector must never be the thing that breaks a turn."""
    from cogno_synapse.openai_backend import _warn_if_truncated

    assert _warn_if_truncated(object(), "gpt-4o-mini") is False
    assert _warn_if_truncated(type("R", (), {"choices": []})(), "gpt-4o-mini") is False


# ── reasoning models that refuse function tools ───────────────────────────────────────

class _ToolResp:
    """A successful tool-calling response."""

    def __init__(self) -> None:
        call = type("TC", (), {"id": "c1", "function": type("F", (), {
            "name": "get_weather", "arguments": '{"city":"SP"}'})()})()
        self.choices = [type("C", (), {
            "finish_reason": "tool_calls",
            "message": type("M", (), {"content": "", "tool_calls": [call]})()})()]
        self.usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 18})()


class _UnsupportedParamError(Exception):
    """The OTHER 400 that also carries ``param='reasoning_effort'``: an endpoint that does not
    accept the parameter at all. Retrying here re-sends what was just rejected."""

    param = "reasoning_effort"

    def __str__(self) -> str:
        return ("Error code: 400 - {'error': {'message': \"Unsupported parameter: "
                "'reasoning_effort' is not supported with this model.\"}}")


class _ReasoningToolsError(Exception):
    """The provider's real 400, verbatim (measured 2026-08-20 on gpt-5.6-luna)."""

    param = "reasoning_effort"

    def __str__(self) -> str:
        return ("Error code: 400 - {'error': {'message': \"Function tools with reasoning_effort "
                "are not supported for gpt-5.6-luna in /v1/chat/completions. To use function "
                "tools, use /v1/responses or set reasoning_effort to 'none'.\"}}")


@pytest.mark.parametrize("exc, matches", [
    (_ReasoningToolsError(), True),
    # message-only (no ``param`` attribute) — a raw wire error still has to match
    (Exception("Function tools with reasoning_effort are not supported for x"), True),
    # BOTH signals required — and this must be the REALISTIC shape: the SDK sets
    # param='reasoning_effort' on a plain "unsupported parameter" 400 too, which is what an
    # OpenAI-compatible endpoint (base_url) returns when it rejects the parameter outright.
    # A bare Exception has no ``param`` and so never reached the branch it was meant to pin.
    (_UnsupportedParamError(), False),
    (Exception("Unsupported parameter: reasoning_effort"), False),
    (Exception("Function tools are not supported for this model"), False),
    (Exception("rate limit exceeded"), False),
])
def test_the_conflict_detector_needs_BOTH_signals(exc, matches):
    from cogno_synapse.openai_backend import _is_reasoning_tools_conflict

    assert _is_reasoning_tools_conflict(exc) is matches


@pytest.mark.asyncio
async def test_a_model_that_refuses_tools_while_reasoning_is_retried_with_effort_none(caplog):
    """Measured 2026-08-20: gpt-5.6-luna/terra/sol return 400 on EVERY tool call, so every EGO
    turn on those models died. gpt-5, gpt-5-mini, gpt-5-nano and the 5.4 family were unaffected
    — the failure is per-model, not "GPT-5 is broken".

    The provider names its own remedy in the error, so the retry follows the instruction rather
    than a hardcoded model list: the constraint belongs to the model, and enumerating names
    means the next model with the same rule breaks in production until someone notices."""
    import logging

    from cogno_synapse.openai_backend import OpenAIBackend

    seen: list = []

    class _Completions:
        async def create(self, **kw):
            seen.append(kw)
            if len(seen) == 1:
                raise _ReasoningToolsError()
            return _ToolResp()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            return None

    b = OpenAIBackend(model="gpt-5.6-luna", api_key="sk-x")
    b._client = lambda: _Client()  # type: ignore[method-assign]

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.openai"):
        msg, tin, tout = await b.chat_with_tools([{"role": "user", "content": "hi"}], tools)

    assert len(msg["tool_calls"]) == 1 and tout == 18
    assert len(seen) == 2, "the call must be retried once, not abandoned"
    assert "reasoning_effort" not in seen[0], "the first attempt sends it as before"
    assert seen[1]["reasoning_effort"] == "none", "the retry applies the provider's remedy"
    assert "reasoning_tools_conflict" in caplog.text


@pytest.mark.asyncio
async def test_an_unrelated_failure_is_not_retried():
    """A retry on every 400 would double the cost of a genuine failure and hide its cause."""
    from cogno_synapse.openai_backend import OpenAIBackend

    calls: list = []

    class _Completions:
        async def create(self, **kw):
            calls.append(kw)
            raise Exception("upstream is on fire")

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            return None

    b = OpenAIBackend(model="gpt-5", api_key="sk-x")
    b._client = lambda: _Client()  # type: ignore[method-assign]

    with pytest.raises(Exception, match="on fire"):
        await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_generate_keeps_full_reasoning():
    """The remedy disables reasoning, so it is applied ONLY where tools are required. NOUMENO,
    NER and the voice go through ``generate`` and must keep the model the tenant chose."""
    from cogno_synapse.openai_backend import OpenAIBackend

    seen: list = []

    class _Completions:
        async def create(self, **kw):
            seen.append(kw)
            raise _ReasoningToolsError()      # the same 400 chat_with_tools recovers from

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            return None

    b = OpenAIBackend(model="gpt-5.6-luna", api_key="sk-x")
    b._client = lambda: _Client()  # type: ignore[method-assign]

    # It must PROPAGATE here, not be recovered: a stub that always succeeds proved nothing —
    # copying the retry into ``generate`` left the old version of this test green.
    with pytest.raises(Exception, match="Function tools"):
        await b.generate("system", "prompt")
    assert len(seen) == 1, "generate must not retry — the remedy disables reasoning"
    assert "reasoning_effort" not in seen[0]


@pytest.mark.asyncio
async def test_the_conflict_is_learned_once_not_rediscovered_every_step():
    """An affected model fails EVERY tool call, and the EGO reuses one backend across 5–8 steps
    plus correction retries. Without memoising, each step pays a full failed round-trip first —
    5–8 wasted 400s per turn, every turn, forever."""
    from cogno_synapse.openai_backend import OpenAIBackend

    seen: list = []

    class _Completions:
        async def create(self, **kw):
            seen.append(dict(kw))
            if "reasoning_effort" not in kw:
                raise _ReasoningToolsError()
            return _ToolResp()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            return None

    b = OpenAIBackend(model="gpt-5.6-luna", api_key="sk-x")
    b._client = lambda: _Client()  # type: ignore[method-assign]
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]

    await b.chat_with_tools([{"role": "user", "content": "a"}], tools)   # 400 + retry
    await b.chat_with_tools([{"role": "user", "content": "b"}], tools)   # must go straight
    await b.chat_with_tools([{"role": "user", "content": "c"}], tools)

    assert len(seen) == 4, f"expected 2 calls then 1 each, got {len(seen)}"
    assert all("reasoning_effort" in c for c in seen[1:])


# ── the prompt cache the provider already gives us and nobody read ────────────────────────
#
# Measured live 2026-09-03: a second call with the same prefix reported cached 2432 of 2625
# prompt tokens (92.6%). `prompt_tokens_details.cached_tokens` was read NOWHERE in the stack,
# so the meter priced every EGO correction retry as a fresh prompt.

class _CachedResp:
    """A successful text response whose usage carries a cached-prompt block."""

    def __init__(self, cached=2432, prompt=2625, details=True) -> None:
        self.choices = [_Choice("stop", "hello")]
        u = type("U", (), {"prompt_tokens": prompt, "completion_tokens": 7})()
        if details is True:
            u.prompt_tokens_details = type("D", (), {"cached_tokens": cached})()
        elif details == "dict":
            u.prompt_tokens_details = {"cached_tokens": cached}
        elif details == "garbage":
            u.prompt_tokens_details = type("D", (), {"cached_tokens": "many"})()
        self.usage = u


def _client_returning(resp, record=None):
    class _Completions:
        async def create(self, **kw):
            if record is not None:
                record.append(kw)
            if isinstance(resp, Exception):
                raise resp
            return resp

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            return None

    return _Client


@pytest.mark.parametrize("details, expected", [
    (True, 2432),          # the shape the OpenAI SDK returns
    ("dict", 2432),        # an OpenAI-COMPATIBLE endpoint handing back a plain dict
    (False, 0),            # no block at all — "unknown", priced at the full rate as before
    ("garbage", 0),        # a non-numeric count must degrade, never raise
])
@pytest.mark.asyncio
async def test_generate_reports_the_cached_prompt_tokens(details, expected):
    from cogno_synapse import cached_tokens_of
    from cogno_synapse.openai_backend import OpenAIBackend

    b = OpenAIBackend(model="gpt-4o-mini", api_key="sk-x")
    b._client = _client_returning(_CachedResp(details=details))  # type: ignore[method-assign]
    text, tin, tout = await b.generate("sys", "hi")
    assert (text, tin, tout) == ("hello", 2625, 7)
    assert cached_tokens_of(b) == expected


@pytest.mark.asyncio
async def test_the_cached_count_is_a_SUBSET_of_the_prompt_tokens_it_is_reported_with():
    """The whole arithmetic downstream is ``fresh = tokens_in - cached``. If this number were
    an EXTRA (the Anthropic shape) rather than a part, the meter would price a negative
    prompt. Pinned here, where the provider's shape is known, not four repos away."""
    from cogno_synapse import cached_tokens_of
    from cogno_synapse.openai_backend import OpenAIBackend

    b = OpenAIBackend(model="gpt-4o-mini", api_key="sk-x")
    b._client = _client_returning(_CachedResp())  # type: ignore[method-assign]
    _, tokens_in, _ = await b.generate("sys", "hi")
    assert 0 < cached_tokens_of(b) <= tokens_in


@pytest.mark.asyncio
async def test_a_call_that_reports_no_cache_does_not_inherit_the_previous_calls_count():
    """The value is per CALL. A stale one is worse than none: it would hand the meter a
    discount for a prompt the provider charged in full."""
    from cogno_synapse import cached_tokens_of
    from cogno_synapse.openai_backend import OpenAIBackend

    b = OpenAIBackend(model="gpt-4o-mini", api_key="sk-x")
    b._client = _client_returning(_CachedResp())  # type: ignore[method-assign]
    await b.generate("sys", "hi")
    assert cached_tokens_of(b) == 2432
    b._client = _client_returning(_CachedResp(details=False))  # type: ignore[method-assign]
    await b.generate("sys", "hi")
    assert cached_tokens_of(b) == 0


@pytest.mark.asyncio
async def test_a_FAILED_call_leaves_no_cached_count_behind():
    """A raise must clear it too, or the next successful call is billed at the previous
    one's discount."""
    from cogno_synapse import cached_tokens_of
    from cogno_synapse.openai_backend import OpenAIBackend

    b = OpenAIBackend(model="gpt-4o-mini", api_key="sk-x")
    b._client = _client_returning(_CachedResp())  # type: ignore[method-assign]
    await b.generate("sys", "hi")
    b._client = _client_returning(RuntimeError("boom"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await b.generate("sys", "hi")
    assert cached_tokens_of(b) == 0


@pytest.mark.asyncio
async def test_chat_with_tools_reports_it_too():
    """The EGO's correction retries are the measured case, and they go through the TOOL path.
    Reporting it only on ``generate`` would miss the money."""
    from cogno_synapse import cached_tokens_of
    from cogno_synapse.openai_backend import OpenAIBackend

    resp = _ToolResp()
    resp.usage = type("U", (), {
        "prompt_tokens": 3874, "completion_tokens": 18,
        "prompt_tokens_details": type("D", (), {"cached_tokens": 3584})()})()
    b = OpenAIBackend(model="gpt-4o-mini", api_key="sk-x")
    b._client = _client_returning(resp)  # type: ignore[method-assign]
    msg, tin, tout = await b.chat_with_tools([{"role": "user", "content": "hi"}], [])
    assert tin == 3874 and len(msg["tool_calls"]) == 1
    assert cached_tokens_of(b) == 3584


def test_a_backend_that_reports_nothing_answers_zero_not_an_error():
    """Ollama, a stub, the distilled student: silence is 0, and 0 downstream means FULL price
    — the old behaviour. The helper must never be the thing that breaks a turn."""
    from cogno_synapse import cached_tokens_of

    assert cached_tokens_of(object()) == 0
    assert cached_tokens_of(type("B", (), {"last_cached_tokens": None})()) == 0
    assert cached_tokens_of(type("B", (), {"last_cached_tokens": -9})()) == 0
    assert cached_tokens_of(type("B", (), {"last_cached_tokens": 12})()) == 12


@pytest.mark.asyncio
async def test_a_fallback_chain_answers_for_the_link_that_actually_ran():
    """``FallbackBackend`` already forwards ``model`` from the successful backend; the cache
    count has to travel the same way or the ledger pairs one backend's model with another
    backend's (or with no) cache count."""
    from cogno_synapse import FallbackBackend, cached_tokens_of

    class _Dead:
        model = "dead"
        last_cached_tokens = 999

        async def generate(self, system, prompt):
            raise RuntimeError("down")

    class _Live:
        model = "gpt-4o-mini"
        last_cached_tokens = 0

        async def generate(self, system, prompt):
            self.last_cached_tokens = 2432
            return "ok", 2625, 7

    chain = FallbackBackend([_Dead(), _Live()])
    assert cached_tokens_of(chain) == 0          # nothing has run yet
    assert await chain.generate("s", "p") == ("ok", 2625, 7)
    assert chain.model == "gpt-4o-mini"
    assert cached_tokens_of(chain) == 2432
