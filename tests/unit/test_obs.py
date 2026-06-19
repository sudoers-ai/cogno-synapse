"""Unit tests for the shared logging helpers (cogno_synapse._obs)."""

import logging

from cogno_synapse._obs import is_retryable, log_done, status_of, warn_if_retryable


class _Err(Exception):
    def __init__(self, status):
        super().__init__("boom")
        self.status_code = status


class _RespErr(Exception):
    """Mimics httpx.HTTPStatusError: status lives on .response.status_code."""
    def __init__(self, status):
        super().__init__("boom")
        self.response = type("R", (), {"status_code": status})()


def test_status_of_reads_direct_and_nested():
    assert status_of(_Err(429)) == 429
    assert status_of(_RespErr(503)) == 503
    assert status_of(Exception("no status")) is None


def test_is_retryable_only_for_429_5xx():
    assert is_retryable(_Err(429)) is True
    assert is_retryable(_RespErr(502)) is True
    assert is_retryable(_Err(400)) is False
    assert is_retryable(_Err(401)) is False
    assert is_retryable(Exception("none")) is False


def test_log_done_emits_info_keyvalue(caplog):
    logger = logging.getLogger("cogno_synapse.test")
    with caplog.at_level(logging.INFO, logger="cogno_synapse.test"):
        log_done(logger, "openai", "gpt-4o", t0=0.0, tokens_in=12, tokens_out=34)
    rec = caplog.records[-1]
    assert rec.levelno == logging.INFO
    assert "event=generate_done provider=openai model=gpt-4o" in rec.message
    assert "tokens_in=12 tokens_out=34" in rec.message


def test_warn_if_retryable_warns_then_silent(caplog):
    logger = logging.getLogger("cogno_synapse.test")
    with caplog.at_level(logging.WARNING, logger="cogno_synapse.test"):
        warn_if_retryable(logger, "groq", "llama", _Err(429))   # → WARNING
        warn_if_retryable(logger, "groq", "llama", _Err(400))   # → nothing
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "event=generate_retryable provider=groq model=llama status=429" in warnings[0].message
