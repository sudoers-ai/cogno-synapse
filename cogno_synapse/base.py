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


@runtime_checkable
class Embedder(Protocol):
    """Protocol for calculating embeddings and semantic similarity."""
    async def embed(self, text: str) -> list[float]:
        """Generates embedding vector for the given text."""
        ...

    async def similarity(self, a: str, b: str) -> float:
        """Calculates cosine similarity between two texts [0.0, 1.0]."""
        ...
