"""One SDK client per backend instance, reused across calls — and per event loop.

Until 0.1.4 every cloud backend built a NEW SDK client (``AsyncOpenAI``, ``AsyncAnthropic``,
``AsyncGroq``) on every call and closed it at the end. Each client owns its own HTTP connection
pool, so every call paid a fresh TCP + TLS handshake, and nothing was ever kept warm.

The client is now created on the first call and reused by every call after it. It is cached
PER EVENT LOOP: an ``httpx`` pool is bound to the loop it was opened on, so a backend that
outlives one loop (a test suite that runs each test under its own ``asyncio.run``) gets a new
client on the next loop instead of a pool whose loop is closed. ``aclose()`` closes the cached
client; a host calls it at shutdown (the process exiting closes the sockets anyway).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional


def _running_loop() -> "Optional[asyncio.AbstractEventLoop]":
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class LoopBoundClient:
    """The cached client of ONE backend instance: built by ``factory`` on first use, and again
    only when the running event loop changed."""

    def __init__(self, factory: "Callable[[], Any]") -> None:
        self._factory = factory
        self._client: Any = None
        self._loop: "Optional[asyncio.AbstractEventLoop]" = None

    def get(self) -> Any:
        loop = _running_loop()
        if self._client is None or self._loop is not loop:
            self._client = self._factory()
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        client, self._client, self._loop = self._client, None, None
        if client is None:
            return
        try:
            await client.close()
        except (TypeError, AttributeError):
            pass  # a test double without an async close
