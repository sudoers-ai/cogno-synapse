# Host Integration Guide

How to wire `cogno-synapse` into a real application. The library ships the
**model transport**; the host owns **which model, which key, and the cost**. This
is the human-facing companion to `examples/host_min.py`.

> TL;DR — pick a backend (object or `"provider:model"` string) and an embedder,
> optionally wrap them in a `FallbackBackend`, and inject them into the cognitive
> stages. Backends **raise** on failure; the host catches/fails over/prices.

---

## 1. The boundary

| Concern | Owner |
| --- | --- |
| Backend protocols + concrete backends + factory + fallback | **synapse** |
| Resilience (circuit breaker / retry / metrics) | **cogno-homeo** (via `FallbackBackend`) |
| Which model per role/tenant, model ladders (`_FALLBACK_MATRIX`) | **host** |
| API keys + rotation (no tenant-key contextvar here) | **host** |
| Token **counting** (raw `tokens_in/out`) | synapse (returned per call) |
| Token **pricing / billing** | **host** |

---

## 2. Backends & the factory

Every backend satisfies `LLMBackend` (`async generate(system, prompt) ->
(text, tokens_in, tokens_out)` + a `model` attr). Cloud backends also satisfy the
optional `ToolCallingBackend` (`chat_with_tools` + `supports_native_tools`) for
native function calling; a plain `LLMBackend` (a stub, the local Ollama default)
uses the text-fallback path (`<TOOL_CALL>` tags, parsed by
`parse_tool_calls_from_text`).

```python
from cogno_synapse import OllamaBackend, create_backend

gen = OllamaBackend(model="mistral:latest")          # local
gen = create_backend("openai:gpt-4o-mini")           # cloud (needs OPENAI_API_KEY)
gen = create_backend("deepseek:deepseek-chat")       # OpenAI-compatible via base_url
```

`create_backend("provider:model")` raises `MissingAPIKeyError` for a cloud
provider without a usable key (fail loudly, never silently degrade to a local
model the caller did not ask for). Supported prefixes: `openai`, `anthropic`,
`groq`, `gemini`, `bedrock`, the OpenAI-compatible set (`deepseek`, `moonshot`,
`xai`, `openrouter`, `together`, `fireworks`), and `ollama` (also the
bare/unknown fallback). There is deliberately **no `mistral:` prefix** — it would
clobber Ollama's `mistral:latest`; `create_backend("mistral:latest")` resolves to
Ollama.

Install only the SDKs you use:

```bash
pip install "cogno-synapse[openai]"     # or [anthropic|groq|gemini|bedrock|llm]
```

## 3. Embedder

```python
from cogno_synapse import OllamaEmbedder, CachingEmbedder

embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text:latest"))
```

`CachingEmbedder` wraps **any** `Embedder` with a bounded LRU (by lowercased text)
plus token/call accounting (`embedder.usage -> EmbeddingUsage`). Used by NOUMENO
(subject continuity) and ID (goal similarity) in `cogno-anima`.

## 4. Resilient failover

`FallbackBackend` tries an ordered chain, first success wins, last error
propagates; it skips non-FC backends for `chat_with_tools`. Its loop runs over
`cogno-homeo`, so pass a breaker/retry/metrics to harden it (or nothing for the
historical behaviour):

```python
from cogno_synapse import FallbackBackend, create_backend
from cogno_homeo import CircuitBreaker, RetryPolicy

chain = FallbackBackend(
    [create_backend("openai:gpt-4o-mini"), create_backend("groq:llama-3.1-8b-instant")],
    breaker=CircuitBreaker(), policy=RetryPolicy(max_retries=2),
)
```

The business model ladder (which models in which order per tenant) is **host**
policy — synapse just runs the chain you build.

## 5. Error contract

Backends **raise** on transport/auth failure rather than returning `("", 0, 0)`:
`InvalidAPIKeyError` (401/403) and `MissingAPIKeyError` are `SynapseError`
subclasses. Auth errors are config problems retry/fallback can't fix, so they're
typed apart from transient failures — surface them distinctly.

## 6. Tokens

`generate` returns `(text, tokens_in, tokens_out)` and `EmbeddingUsage` tracks
embedding tokens — synapse **produces** the counts. Aggregation per turn happens
in `cogno-anima` (`StageMetrics`/`PipelineContext`); **pricing/billing is the
host's**.
