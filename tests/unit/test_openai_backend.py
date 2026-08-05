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
