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
