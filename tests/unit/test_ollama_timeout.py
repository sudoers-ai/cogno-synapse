"""The Ollama call timeout, and why it is env-steerable.

120 s is right for a GPU box. A CPU-only host — a small VM, a GitHub runner — can take
longer than that for ONE generation on an 8B model, and dies on `httpx.ReadTimeout` with
nothing wrong but the clock. That is what turned the nightly Ollama canaries in
`cogno-anima` and `cogno-soma` red on 2026-08-04: 40 of 41 tests passed and the run was
still a failure.

Read per construction rather than at import, so a test or a host can set the variable after
this module is already loaded.
"""

from __future__ import annotations

import pytest

from cogno_synapse.ollama import _DEFAULT_TIMEOUT, OllamaBackend, OllamaEmbedder, default_timeout


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv("COGNO_OLLAMA_TIMEOUT", raising=False)
    assert default_timeout() == _DEFAULT_TIMEOUT


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("COGNO_OLLAMA_TIMEOUT", "600")
    assert default_timeout() == 600


def test_read_per_construction_not_at_import(monkeypatch):
    """The whole point: the module is imported long before CI sets the variable."""
    monkeypatch.setenv("COGNO_OLLAMA_TIMEOUT", "300")
    assert OllamaBackend(model="m").timeout == 300
    assert OllamaEmbedder().timeout == 300


def test_explicit_argument_still_wins(monkeypatch):
    monkeypatch.setenv("COGNO_OLLAMA_TIMEOUT", "600")
    assert OllamaBackend(model="m", timeout=42).timeout == 42
    assert OllamaEmbedder(timeout=42).timeout == 42


@pytest.mark.parametrize("bad", ["", "   ", "abc", "0", "-5", "12.5"])
def test_a_bad_value_falls_back_instead_of_raising(monkeypatch, bad):
    """A typo'd env var must not take the process down, and must not mean 'no timeout'."""
    monkeypatch.setenv("COGNO_OLLAMA_TIMEOUT", bad)
    assert default_timeout() == _DEFAULT_TIMEOUT
    assert OllamaBackend(model="m").timeout == _DEFAULT_TIMEOUT
