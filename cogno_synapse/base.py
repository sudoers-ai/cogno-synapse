from typing import Optional, Protocol, runtime_checkable

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


def _last_provider_string(backend: object, attribute: str) -> Optional[str]:
    """Read a per-call identifier the provider stamped on the LAST response, or ``None``.

    The single normaliser behind ``system_fingerprint_of`` and ``served_model_of``, so both
    answer "the provider did not say" the same way. Anything that is not a non-blank string —
    a missing attribute, ``None``, an empty or whitespace-only echo, a number — reads as
    ``None``: these values are only ever COMPARED to one another, and a blank that compares
    equal to another blank would assert that two calls were served by the same thing when
    neither said anything at all.
    """
    value = getattr(backend, attribute, None)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def system_fingerprint_of(backend: object) -> Optional[str]:
    """The provider's identifier for the backend configuration that served the LAST call.

    THE single definition, like ``cached_tokens_of`` above — a consumer asks here instead of
    reaching for the attribute, so "which backends report it, and what does silence mean" is
    answered in one place.

    **What it is for.** At ``temperature=0`` a hosted provider only *requests* greedy decoding;
    it does not promise it, and the snapshot behind a model ALIAS can move without the name
    changing. So a byte-identical prompt can be classified one way at noon and the other way at
    one, and from outside there is nothing to compare. OpenAI already answers that question on
    every response and this layer was discarding it: with the fingerprint recorded per call,
    "did the backend change under us?" is a string comparison rather than a theory.

    **``None`` means the provider did not say** — it is never ``""`` and never a stand-in
    constant. A backend with no such notion, a provider that omits the field, and a call that
    has not happened yet all answer ``None``, and ``None`` must stay distinguishable from every
    real fingerprint: two calls that both answer ``None`` are two calls that told us nothing,
    NOT two calls served by the same configuration.

    **Read it IMMEDIATELY after the ``await`` that produced the call**, with no other ``await``
    in between::

        text, tin, tout = await backend.generate(system, prompt)
        fingerprint = system_fingerprint_of(backend)   # ← no suspension point above this line

    Same property, and the same limits, as ``cached_tokens_of``: a per-INSTANCE last-call
    attribute is safe on a backend shared between concurrent turns only because a
    single-threaded event loop cannot run another coroutine between the await resolving and the
    next statement. It is NOT safe across an ``await``, and not safe if a caller drives the same
    instance from two OS threads. A per-call return channel would need a change to the
    ``generate``/``chat_with_tools`` tuples that every stage, stub and downstream backend
    satisfies; this does not.

    Who reports one:

      * **OpenAI** — ``response.system_fingerprint``, on both the text and the tool-calling
        path. Reported.
      * **The OpenAI-COMPATIBLE endpoints** reached through ``OpenAIBackend(base_url=...)``
        (DeepSeek, Moonshot/Kimi, xAI, OpenRouter, Together, Fireworks) — most omit the field.
        Read defensively; a missing one is ``None``, never an exception.
      * **Groq** — OpenAI-shaped response, same field, read the same way. Reported when present.
      * **Anthropic / Gemini / Bedrock / Ollama** — no equivalent field exists. ``None``. An
        invented substitute (a model name, a hash of the request) would compare equal across
        genuinely different backends, which is the one failure this must not have.
    """
    return _last_provider_string(backend, "last_system_fingerprint")


def served_model_of(backend: object) -> Optional[str]:
    """The model id the provider ECHOED BACK on the LAST call, or ``None``.

    The other half of the same question. ``backend.model`` is what we ASKED for and may be an
    alias (``gpt-4o-mini``); this is what answered, and for OpenAI that is the dated snapshot
    (``gpt-4o-mini-2024-07-18``). An alias moving to a new snapshot and a replica running a
    different backend configuration are two different drifts, and only these two fields
    together tell them apart.

    Identical rules to ``system_fingerprint_of``: ``None`` when unknown, read immediately after
    the ``await``, forwarded by ``FallbackBackend`` from the link that actually ran.
    """
    return _last_provider_string(backend, "last_served_model")


@runtime_checkable
class Embedder(Protocol):
    """Protocol for calculating embeddings and semantic similarity."""
    async def embed(self, text: str) -> list[float]:
        """Generates embedding vector for the given text."""
        ...

    async def similarity(self, a: str, b: str) -> float:
        """Calculates cosine similarity between two texts [0.0, 1.0]."""
        ...
