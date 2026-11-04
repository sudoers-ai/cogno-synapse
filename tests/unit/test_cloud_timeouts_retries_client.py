"""A cloud call gives up at its timeout, retries in ONE layer, and reuses its client (0.1.4).

The defect, measured downstream: a turn of 1009 s. The factory gave every backend 600 s, the SDK
retried twice UNDER the host's retry policy, and every call built a new `AsyncOpenAI` (a new
connection, a new handshake). The proofs here run against a fake HTTP server on the loopback
interface. It counts the TCP connections it accepts and the requests it receives, and it can
answer, fail with a 500, or never answer. Nothing reaches a provider.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import types

import pytest

openai = pytest.importorskip("openai")

from cogno_homeo import RetryPolicy  # noqa: E402

from cogno_synapse import FallbackBackend, create_backend  # noqa: E402
from cogno_synapse.openai_backend import OpenAIBackend  # noqa: E402
from cogno_synapse.timeouts import DEFAULT_CLOUD_TIMEOUT_S, cloud_timeout  # noqa: E402

_COMPLETION = json.dumps({
    "id": "chatcmpl-x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()


class FakeServer:
    """HTTP/1.1 with keep-alive on 127.0.0.1. ``mode``: ``ok`` | ``fail`` (500) | ``hang``."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.connections = 0
        self.requests = 0
        self._server: "asyncio.base_events.Server | None" = None
        # Python 3.12's `Server.wait_closed()` waits for every open connection to close
        # (3.10/3.11 returned at once), so a ``hang`` handler left sleeping held the test's
        # teardown forever. Exiting releases the hang, and the handler closes its connection.
        self._release = asyncio.Event()

    async def __aenter__(self) -> "FakeServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._server is not None
        self._release.set()
        self._server.close()
        await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1])
                if length:
                    await reader.readexactly(length)
                self.requests += 1
                if self.mode == "hang":
                    await self._release.wait()          # until the server exits — never answers
                    return
                if self.mode == "fail":
                    body, status = b'{"error": {"message": "boom"}}', b"500 Internal Server Error"
                else:
                    body, status = _COMPLETION, b"200 OK"
                writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
                             b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                             b"Connection: keep-alive\r\n\r\n" + body)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            writer.close()


def _backend(server: FakeServer, **kw) -> OpenAIBackend:
    return OpenAIBackend(model="m", api_key="sk-test-not-a-key", base_url=server.base_url, **kw)


# ── the client is reused ──

@pytest.mark.asyncio
async def test_GEMEO_n_calls_of_one_backend_open_one_connection():
    async with FakeServer() as server:
        b = _backend(server, timeout=5)
        for _ in range(5):
            assert (await b.generate("s", "p"))[0] == "ok"
        assert server.requests == 5
        assert server.connections == 1
        await b.aclose()


@pytest.mark.asyncio
async def test_CONTROLO_three_backends_open_three_connections_and_aclose_releases_one():
    """The counter counts: separate instances, separate clients."""
    async with FakeServer() as server:
        backends = [_backend(server, timeout=5) for _ in range(3)]
        for b in backends:
            await b.generate("s", "p")
        assert server.connections == 3
        await backends[0].aclose()                       # closed → the next call reconnects
        await backends[0].generate("s", "p")
        assert server.connections == 4
        for b in backends:
            await b.aclose()


def test_a_new_event_loop_gets_a_new_client():
    """An httpx pool is bound to its loop: a backend used under two `asyncio.run`s must not
    hand the second loop a pool of the first."""
    b = OpenAIBackend(model="m", api_key="sk-test-not-a-key")

    async def grab():
        return b._client()
    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second

    async def twice():
        return b._client() is b._client()
    assert asyncio.run(twice()) is True                  # CONTROL: one loop, one client


# ── the timeout, and one retry layer ──

@pytest.mark.asyncio
async def test_GEMEO_a_server_that_never_answers_is_cut_at_the_timeout_with_one_request():
    async with FakeServer(mode="hang") as server:
        b = _backend(server, timeout=0.5)
        t0 = time.monotonic()
        with pytest.raises(openai.APITimeoutError):
            await b.generate("s", "p")
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, elapsed                    # cut at ~0.5 s, never 600
        assert server.requests == 1                      # the SDK did not retry
        await b.aclose()


@pytest.mark.asyncio
async def test_GEMEO_a_failing_provider_is_asked_exactly_the_homeo_attempts():
    async with FakeServer(mode="fail") as server:
        b = _backend(server, timeout=5)
        chain = FallbackBackend([b], policy=RetryPolicy(max_retries=2, base_ms=1.0, max_ms=1.0,
                                                        jitter=False))
        with pytest.raises(Exception):
            await chain.generate("s", "p")
        assert server.requests == 3                      # 1 + 2 homeo retries, 0 from the SDK
        await chain.aclose()


def test_the_sdk_client_is_built_without_retries_and_with_the_30_s_timeout(monkeypatch):
    """The default reaches the SDK client itself — without waiting 30 real seconds (the cut is
    measured above at 0.5 s)."""
    monkeypatch.delenv("COGNO_CLOUD_TIMEOUT_S", raising=False)

    async def build():
        return OpenAIBackend(model="m", api_key="sk-test-not-a-key")._client()
    client = asyncio.run(build())
    assert client.max_retries == 0
    assert client.timeout == 30.0


# ── the defaults ──

def test_the_factory_gives_a_cloud_backend_30_s_and_ollama_600(monkeypatch):
    monkeypatch.delenv("COGNO_CLOUD_TIMEOUT_S", raising=False)
    assert DEFAULT_CLOUD_TIMEOUT_S == 30.0
    assert create_backend("openai:gpt-4o-mini", api_key="sk-x").timeout == 30.0
    assert create_backend("deepseek:deepseek-chat", api_key="sk-x").timeout == 30.0
    assert create_backend("openai:gpt-4o-mini", api_key="sk-x", timeout=7).timeout == 7
    assert create_backend("ollama:qwen3:8b").timeout == 600          # unchanged on purpose
    assert OpenAIBackend(model="m", api_key="sk-x").timeout == 30.0  # built directly, too


@pytest.mark.parametrize("raw, expected", [("12", 12.0), ("2.5", 2.5), ("", 30.0),
                                           ("abc", 30.0), ("0", 30.0), ("-4", 30.0)],
                         ids=["int", "float", "empty", "garbage", "zero", "negative"])
def test_the_environment_steers_the_cloud_timeout(monkeypatch, raw, expected):
    monkeypatch.setenv("COGNO_CLOUD_TIMEOUT_S", raw)
    assert cloud_timeout() == expected
    assert create_backend("openai:gpt-4o-mini", api_key="sk-x").timeout == expected


# ── Anthropic and Groq: the same client rule (their SDKs are optional, so a fake module) ──

class _FakeSDKClient:
    built: "list[dict]" = []

    def __init__(self, **kwargs) -> None:
        type(self).built.append(kwargs)

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("which", ["anthropic", "groq"])
def test_anthropic_and_groq_build_one_client_without_retries(monkeypatch, which):
    _FakeSDKClient.built = []
    if which == "anthropic":
        monkeypatch.setitem(sys.modules, "anthropic",
                            types.SimpleNamespace(AsyncAnthropic=_FakeSDKClient))
        from cogno_synapse.anthropic_backend import AnthropicBackend as Backend
    else:
        monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(AsyncGroq=_FakeSDKClient))
        from cogno_synapse.groq_backend import GroqBackend as Backend
    monkeypatch.delenv("COGNO_CLOUD_TIMEOUT_S", raising=False)
    b = Backend(model="m", api_key="k")

    async def twice():
        return b._client() is b._client()
    assert asyncio.run(twice()) is True
    assert len(_FakeSDKClient.built) == 1
    assert _FakeSDKClient.built[0]["max_retries"] == 0
    assert _FakeSDKClient.built[0]["timeout"] == 30.0
