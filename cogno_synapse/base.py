from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMBackend(Protocol):
    """Protocol that any LLM client (OpenAI, Ollama, Bedrock, etc.) must implement."""
    model: str

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        """
        Executes a generation call to the LLM.
        Returns a tuple: (response_text, tokens_in, tokens_out)
        """
        ...


@runtime_checkable
class ToolCallingBackend(LLMBackend, Protocol):
    """Optional extension: backends with native function calling (OpenAI,
    Anthropic, Bedrock, Gemini, Groq, Ollama-/api/chat).

    Kept SEPARATE from ``LLMBackend`` on purpose: a text-only backend — a test
    stub, or the distilled student model — implements just ``LLMBackend`` and
    the EGO auto-uses the text-fallback path (``isinstance(backend,
    ToolCallingBackend)`` is False). Putting these methods on ``LLMBackend``
    would force every backend (and NOUMENO/NER/ID, which never call tools) to
    carry them.

    Unlike the parent's "never raise" contract, implementations here RAISE on
    transport/auth failure (errors propagate; the host decides retry/swap) —
    they do not return an empty result.
    """

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: "str | dict | None" = None,
    ) -> tuple[dict, int, int]:
        """Send a multi-turn conversation with native function calling.

        ``messages``/``tools`` are in OpenAI format; backends with a different
        wire format (Anthropic, Gemini) convert internally. Returns
        ``(message_dict, tokens_in, tokens_out)`` where ``message_dict`` is
        ``{"content": str, "tool_calls": [{"id","type","function":{"name","arguments"}}]}``
        (``tool_calls`` absent/empty when the model answers with text).
        """
        ...

    def supports_native_tools(self) -> bool:
        """True if native FC is available right now (the EGO uses it then)."""
        ...


def cached_tokens_of(backend: object) -> int:
    """How many of the LAST call's prompt tokens the provider served from its own cache.

    THE single definition — every consumer (a stage building its ``StageMetrics``, a host
    metering a turn) calls this instead of reaching for the attribute, so "which backends
    report it, and what does silence mean" is answered in one place. A backend that reports
    nothing answers 0, and 0 means *unknown or none*: downstream that is priced at the full
    input rate, which is the old behaviour and the safe direction.

    **Read it IMMEDIATELY after the ``await`` that produced the call**, with no other ``await``
    in between::

        text, tin, tout = await backend.generate(system, prompt)
        cached = cached_tokens_of(backend)          # ← no suspension point above this line

    That is what makes a per-instance value safe on a backend shared between concurrent turns:
    under a single-threaded event loop no other coroutine can run between the await resolving
    and the next statement. It is NOT safe across an ``await``, and it is not safe if a caller
    drives the same backend instance from two OS threads.

    Only the subset that is genuinely a subset. ``LLMBackend.generate`` reports ``tokens_in``,
    and this number must be part of it or the arithmetic downstream (``fresh = tokens_in -
    cached``) is wrong:

      * **OpenAI** (and the OpenAI-compatible endpoints reached through ``base_url``) —
        ``prompt_tokens`` INCLUDES ``prompt_tokens_details.cached_tokens``. Reported.
      * **Anthropic** — ``input_tokens`` EXCLUDES ``cache_read_input_tokens``, and the write
        side is billed at a SURCHARGE. Not reported: the number would not be a subset, and one
        field cannot carry a read and a write with opposite signs.
      * **Gemini / Bedrock / Groq** — not reported here; nothing has been verified about the
        shape of their counters, and a wrong subset is worse than a missing one.
    """
    return max(0, int(getattr(backend, "last_cached_tokens", 0) or 0))


@runtime_checkable
class Embedder(Protocol):
    """Protocol for calculating embeddings and semantic similarity."""
    async def embed(self, text: str) -> list[float]:
        """Generates embedding vector for the given text."""
        ...

    async def similarity(self, a: str, b: str) -> float:
        """Calculates cosine similarity between two texts [0.0, 1.0]."""
        ...
