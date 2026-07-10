# cogno-synapse

**Model-transport layer for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — LLM + embedding backend protocols, Ollama + cloud backends, a single-backend factory, and a resilient fallback chain.

Named for the *synapse*: the junction that carries the signal between neurons. Where [`cogno-anima`](https://github.com/sudoers-ai/cogno-anima) is the *mind* and [`cogno-engram`](https://github.com/sudoers-ai/cogno-engram) is the *memory*, `cogno-synapse` is the **channel the mind speaks to language/embedding models through**.

> Status: **alpha** — extracted from `cogno-anima`'s `llm/` package; unit suite (mocked SDKs) in place.

## Protocol first, implementation optional

The light, stable contract everyone shares is a set of structurally-typed `Protocol`s (no inheritance required):

- `LLMBackend` — `async generate(system, prompt) -> (text, tokens_in, tokens_out)` + a `model` attr.
- `ToolCallingBackend` (extends `LLMBackend`) — adds `chat_with_tools(...)` + `supports_native_tools()` for native function calling. **Separate and optional**, so a text-only backend (a stub, a distilled student) satisfies only `LLMBackend`.
- `Embedder` — `async embed(text) -> list[float]`, `async similarity(a, b) -> float`.

The heavy concrete cloud backends are **optional extras** (lazy-imported SDKs) — depend on the protocol, install only the providers you use:

```bash
pip install cogno-synapse                       # protocols + Ollama + OpenAI-compatible HTTP + fallback (httpx only)
pip install "cogno-synapse[openai]"             # + OpenAI SDK
pip install "cogno-synapse[anthropic|groq|gemini|bedrock]"
pip install "cogno-synapse[llm]"                # all cloud SDKs
```

## Backends

`OllamaBackend`/`OllamaEmbedder` (local, `think=false` by default so reasoning models still return direct JSON), plus `OpenAIBackend`, `AnthropicBackend`, `GroqBackend`, `GeminiBackend`, `BedrockBackend` — each implements `LLMBackend` + `ToolCallingBackend`. OpenAI-compatible providers (DeepSeek, Moonshot, xAI/Grok, OpenRouter, Together, Fireworks) reuse `OpenAIBackend` via `base_url`; `create_backend("provider:model")` instantiates one by string. Backends **raise** on transport/auth failure (`InvalidAPIKeyError` for 401/403) rather than degrading silently.

`CachingEmbedder` wraps any `Embedder` with a bounded LRU + token accounting (`EmbeddingUsage`). `parse_tool_calls_from_text` reads `<TOOL_CALL>` tags for the text-fallback function-calling path (and rescues FC leaks).

## Resilient fallback — over `cogno-homeo`

`FallbackBackend` tries an ordered chain, first success wins, last error propagates. The loop runs on [`cogno-homeo`](https://github.com/sudoers-ai/cogno-homeo)'s `resilient_call`, so you can opt into a circuit breaker, retry/backoff, and a metrics seam — with none supplied it behaves like the historical "try each once" chain:

```python
from cogno_synapse import FallbackBackend, create_backend
from cogno_homeo import CircuitBreaker, RetryPolicy

chain = FallbackBackend(
    [create_backend("openai:gpt-4o-mini"), create_backend("groq:llama-3.1-8b-instant")],
    breaker=CircuitBreaker(), policy=RetryPolicy(max_retries=2),   # optional
)
text, tin, tout = await chain.generate(system, prompt)
```

## Scope

This library is **transport only** — it produces raw token counts but does not price them (the host does), and it carries no business model-ladder/`_FALLBACK_MATRIX` (also host). Resilience is delegated to `cogno-homeo`; cognition lives in `cogno-anima`.

## The Cogno ecosystem

`cogno-synapse` is one organ of **[Cogno](https://github.com/sudoers-ai)** — a family of
small, composable, Apache-2.0 libraries that together form a complete
conversational-agent platform. Each library owns a single concern and stays
infra-agnostic; a **host** assembles them into a running agent:

![The Cogno ecosystem](docs/assets/cogno-ecosystem.svg)

The open-source libraries are the organs; the **host is the body** that joins
them. Our reference host — `cogno-host`, with its `cogno-ui` dashboard — is the
private product layer, but it holds no special powers: everything it does rides
on the public seams documented in each library's `docs/HOST_INTEGRATION.md`, so
you can assemble a body of your own.

## Test

```bash
pip install -e ".[dev]"      # unit tests mock every SDK — no cloud keys needed
pytest tests/unit -q
```
