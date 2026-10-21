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
model the caller did not ask for). `api_key=` overrides the environment variable
(a per-tenant key, BYOK); a variable that is empty, `dummy`, or contains
`sk-proj-*` counts as missing. Supported prefixes: `openai`, `anthropic`,
`groq`, `gemini`, `bedrock`, the OpenAI-compatible set (`deepseek`, `moonshot`
and its alias `kimi`, `xai` and its alias `grok`, `openrouter`, `together`,
`fireworks`), and `ollama` (also the bare/unknown fallback; an empty string
builds Ollama `llama3.2`). There is deliberately **no `mistral:` prefix** — it would
clobber Ollama's `mistral:latest`; `create_backend("mistral:latest")` resolves to
Ollama. `OllamaBackend` itself is a plain `LLMBackend` (it has no
`chat_with_tools`), which is why it takes the text-fallback path below.

**Timeouts.** `create_backend` passes `timeout=600` and `create_embedder`
`timeout=120` unless told otherwise. `$COGNO_OLLAMA_TIMEOUT` (default 120 s) is
read only by an `OllamaBackend`/`OllamaEmbedder` built WITHOUT a `timeout=`, so
it does not reach one built through the factories.

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

The cloud embedders (`OpenAIEmbedder`, `GeminiEmbedder`, `BedrockEmbedder`) are
built by `create_embedder("provider:model", dimensions=…)`, the same grammar as
`create_backend`. It returns ONE embedder, never a chain — failing over mid-write
mixes embedding spaces in one column — and refuses any provider outside ollama,
openai, gemini and bedrock with `NotImplementedError` at construction.
`dimensions` is the width your vector store was built for; Ollama ignores it.

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

### A schema with the request (`StructuredBackend`)

A third optional protocol, probed like `ToolCallingBackend`. A backend that has
BOTH `schema_dialect(form)` and `generate_structured(system, prompt, *, schema,
name, form)` can send a JSON Schema with the request; `generate` is unchanged for
everyone.

| What | Rule |
| --- | --- |
| Probe | `schema_dialect_of(backend, form)` — the protocol AND a non-`None` answer from the closed `SCHEMA_ANNOUNCEMENTS`. Never raises. |
| Form | `"open"` or `"strict"` (`SCHEMA_FORMS`), per CALL, no default. |
| Schema | The canonical one. The leaf converts (`to_dialect`); a chain passes it on untouched. |
| Receipt | `schema_applied_of(backend)` right after the `await`: a dialect, `"refused"`, or `None`. |
| A 400 on the schema | One repeat on the same backend without it; receipt `"refused"`; remembered per `(model, schema digest)` for 15 minutes. Not seen by the failover loop. |
| 429 / timeout / 5xx | Propagate as from `generate`. |

**If you write a wrapper around a backend**, it is a sink: either define both
members on the CLASS and forward the schema to the backend that serves the call
(reading its receipt on the line after the `await` and exposing it as
`last_schema_dialect`), or define neither and stay a plain `LLMBackend`. Do not
forward through `__getattr__`: from Python 3.12 a `runtime_checkable` protocol
resolves members with `inspect.getattr_static`, which never calls it, so the
wrapper leaves the probe and the schema is dropped without a word.
`FallbackBackend` is the worked example; the per-backend table is in the README.

A chain answers `"chain"` when ANY link answers the form — not all, unlike the
native-tools announcement. The tool announcement picks the prompt, so one link
that cannot keep it breaks the turn; a schema travels outside the prompt, so a
link without a dialect serves the same call through `generate`.

## 5. Error contract

Backends **raise** on transport/auth failure rather than returning `("", 0, 0)`:
`InvalidAPIKeyError` (401/403) and `MissingAPIKeyError` are `SynapseError`
subclasses. Auth errors are config problems retry/fallback can't fix, so they're
typed apart from transient failures — surface them distinctly.

`MissingAPIKeyError` comes from the two factories; a backend constructed directly
without its key only logs a WARNING. The third subclass, `EmbeddingResponseError`,
is raised by `OpenAIEmbedder` when the provider answered but the response cannot
be mapped back to the inputs — returning empty vectors instead would write silent
corruption into a store.

## 6. Tokens

`generate` returns `(text, tokens_in, tokens_out)` and `EmbeddingUsage` tracks
embedding tokens — synapse **produces** the counts. Aggregation per turn happens
in `cogno-anima` (`StageMetrics`/`PipelineContext`); **pricing/billing is the
host's**.

Three more facts about the LAST call are read per instance, each **immediately
after the `await`** that produced it, with no other `await` in between:
`cached_tokens_of(backend)` (the subset of `tokens_in` the provider served from its
own cache — OpenAI and the `base_url` endpoints that send it; 0 = unknown),
`system_fingerprint_of(backend)` and `served_model_of(backend)` (who served the
call — OpenAI and Groq; `None` = not said). A `FallbackBackend` forwards them from
the link that served the call. Price `cached_tokens` apart from the fresh input;
compare the other two between calls, never sum them.
