# cogno-synapse

**Model-transport layer for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — LLM + embedding backend protocols, Ollama + cloud backends, a single-backend factory, and a resilient fallback chain.

Named for the *synapse*: the junction that carries the signal between neurons. Where [`cogno-anima`](https://github.com/sudoers-ai/cogno-anima) is the *mind* and [`cogno-engram`](https://github.com/sudoers-ai/cogno-engram) is the *memory*, `cogno-synapse` is the **channel the mind speaks to language/embedding models through**.

> Status: **alpha** — extracted from `cogno-anima`'s `llm/` package; a unit suite (mocked SDKs and HTTP) plus an integration suite that runs only against a reachable Ollama or real provider keys.

## Protocol first, implementation optional

The light, stable contract everyone shares is a set of structurally-typed `Protocol`s (no inheritance required):

- `LLMBackend` — `async generate(system, prompt) -> (text, tokens_in, tokens_out)` + a `model` attr.
- `ToolCallingBackend` (extends `LLMBackend`) — adds `chat_with_tools(...)` + `supports_native_tools()` for native function calling. **Separate and optional**, so a text-only backend (a stub, a distilled student) satisfies only `LLMBackend`.
- `StructuredBackend` (extends `LLMBackend`) — adds `schema_dialect(form)` + `generate_structured(system, prompt, *, schema, name, form)` for a JSON Schema sent WITH the request. **Separate and optional** for the same reason; `generate` keeps its two arguments. See [A schema with the request](#a-schema-with-the-request).
- `Embedder` — `async embed(text) -> list[float]`, `async similarity(a, b) -> float`.

The heavy concrete cloud backends are **optional extras** (lazy-imported SDKs) — depend on the protocol, install only the providers you use:

```bash
pip install cogno-synapse                       # protocols, Ollama, GeminiEmbedder (REST), fallback, key probe (httpx + cogno-homeo)
pip install "cogno-synapse[openai]"             # + OpenAI SDK — also needed by the OpenAI-compatible providers
pip install "cogno-synapse[anthropic|groq|gemini|bedrock]"
pip install "cogno-synapse[llm]"                # all cloud SDKs
```

The base install depends on `httpx` and `cogno-homeo` (`pyproject.toml`). The OpenAI-compatible providers are served by `OpenAIBackend`, which builds its client from the `openai` SDK, so they need the `openai` extra too. `GeminiEmbedder` calls the REST API over `httpx` and needs no extra; `GeminiBackend` needs `google-genai`.

## Backends

`OpenAIBackend`, `AnthropicBackend`, `GroqBackend`, `GeminiBackend` and `BedrockBackend` each implement `LLMBackend` + `ToolCallingBackend` (native function calling). `OllamaBackend` (local, `/api/generate`, `think=false` by default so reasoning models still return direct JSON) implements `LLMBackend` and `StructuredBackend` but **not** `ToolCallingBackend` — it has no `chat_with_tools`, so a caller uses the text-fallback path with it. OpenAI-compatible providers (DeepSeek, Moonshot/Kimi, xAI/Grok, OpenRouter, Together, Fireworks) reuse `OpenAIBackend` via `base_url`; `create_backend("provider:model")` instantiates one by string (`cogno_synapse/factory.py`, `_OPENAI_COMPATIBLE`). Backends **raise** on transport/auth failure (`InvalidAPIKeyError` for 401/403) rather than degrading silently.

`parse_tool_calls_from_text` reads `<TOOL_CALL>` tags for the text-fallback function-calling path (and rescues FC leaks). Its bracket form (`[tool]`, `[tool(args)]`) counts only when the tag is the only content on its line, so a bracket echoed inside prose — a tool result quoted back — never becomes a call.

Three things `OpenAIBackend` does that a caller would otherwise not see (`cogno_synapse/openai_backend.py`):

- **JSON mode follows the system prompt.** When the `system` text contains "json", `generate` sends `response_format={"type": "json_object"}`. Only the system prompt is read — never `prompt`, which can quote the contact. It is skipped on any `base_url` endpoint and on the reasoning series (`o1`/`o3`/`o4`/`gpt-5`).
- **`reasoning_effort` defaults to sending nothing.** Set it per instance or deployment-wide with `$OPENAI_REASONING_EFFORT`; it is only sent to the reasoning series. When the provider refuses function tools while reasoning is on (a 400 naming both), `chat_with_tools` repeats the call once with `reasoning_effort="none"` — the remedy the error itself names — and the instance sends `"none"` with tools from then on.
- **A cut reply is logged.** A `finish_reason` of `length` or `content_filter` logs `event=truncated_response`; the text still travels on.

### Embedders

`OllamaEmbedder` (local, `/api/embed`, falling back to the legacy `/api/embeddings` once per instance on a 404/405 — that endpoint reports no token count), `OpenAIEmbedder`, `GeminiEmbedder` (REST, the key in the `x-goog-api-key` header) and `BedrockEmbedder` (Titan and Cohere) implement `Embedder`. `create_embedder("provider:model", dimensions=…)` builds ONE of them, never a chain — failing over mid-write would mix embedding spaces in one column. The set is closed (ollama, openai, gemini, bedrock): any other cloud prefix raises `NotImplementedError` at construction. `OpenAIEmbedder` raises `EmbeddingResponseError` when the response cannot be mapped back to its inputs, instead of returning empty vectors.

`CachingEmbedder` wraps any `Embedder` with a bounded LRU (keyed by stripped, lower-cased text; `cache_size=0` disables it) + token accounting (`EmbeddingUsage`), and proxies the batch path.

## What the last call reports besides its text

Three per-instance, last-call readers, each read **immediately after the `await`** that produced the call (`cogno_synapse/base.py`):

| reader | what it answers | who reports it |
|---|---|---|
| `cached_tokens_of(backend)` | the part of `tokens_in` the provider served from its own prompt cache; 0 = unknown or none | OpenAI and the `base_url` endpoints that send the field |
| `system_fingerprint_of(backend)` | the provider's id for the configuration that served the call; `None` = not said | OpenAI, Groq |
| `served_model_of(backend)` | the model id the provider echoed back (the dated snapshot behind an alias); `None` = not said | OpenAI, Groq |

`FallbackBackend` forwards all three from the link that actually served the call. Anthropic's cache counters are not reported: its `input_tokens` excludes the cached part, so the number would not be a subset.

## BYOK — is this key live?

`probe_api_key(provider, api_key)` answers the one question a bring-your-own-key console has to
answer the moment a key is pasted, with a cost-free authenticated GET (list-models, or `/user`)
in that provider's own auth dialect:

```python
from cogno_synapse import API_KEY_PROBES, probe_api_key

live = await probe_api_key("anthropic", key)                      # shipped table
live = await probe_api_key("myprovider", key,                     # …or your own
                           probes={**API_KEY_PROBES, "myprovider": (url, "bearer")})
```

The bias is **fail-open**, and the line it draws is *did the provider make a statement about
this credential?* Only a 401 and a 403 do — the verdict is `status_code not in (401, 403)`.
Everything else is inconclusive and reads as **valid**: a timeout, DNS down, a provider the
table cannot probe, and equally a 429, a 5xx or a 404 from a provider we did reach (those are
logged as `event=byok_probe_inconclusive`, so a rotted probe URL is fail-open but not silent).
A 429 is a rate limit, a 5xx is their infrastructure, a 404 is *our* URL — none is evidence
about the key, and a wrongly-invalidated key locks a paying user out of their own models. A
*blank* key is never trusted. `API_KEY_PROBES` is exported so the caller can pin its own
provider list against it; which providers you offer is yours to decide, not this library's.

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

## A schema with the request

Some providers take a JSON Schema with the request and constrain the reply to it. A backend that can do that satisfies `StructuredBackend`; one that cannot is called through `generate` exactly as before. Nothing calls this by itself — a caller opts in per call:

```python
from cogno_synapse import schema_applied_of, schema_dialect_of

if schema_dialect_of(backend, "strict"):                 # the probe: protocol AND an answer for this form
    text, tin, tout = await backend.generate_structured(
        system, prompt, schema=reply_schema, name="reply", form="strict")
    applied = schema_applied_of(backend)                 # the receipt — read on the line after the await
else:
    text, tin, tout = await backend.generate(system, prompt)
```

- **The form travels with the call**: `"open"` or `"strict"` (`SCHEMA_FORMS`), with **no default** on `schema_dialect` or `generate_structured`. One instance serves many callers, so the form cannot live on the instance.
- **Form → dialect, per backend, and never a demotion** (`cogno_synapse.schema.DIALECT_FOR_FORM`):

  | backend | `form="open"` | `form="strict"` |
  |---|---|---|
  | `OpenAIBackend` (no `base_url`) | `openai_json_schema` | `openai_strict` |
  | `OllamaBackend` | `ollama_format` | `None` |
  | `AnthropicBackend`, `GeminiBackend`, `GroqBackend`, `BedrockBackend`, any `base_url` endpoint | `None` | `None` |
  | `FallbackBackend` | `"chain"` if a link answers that form | same |

  A backend with no dialect for the form asked answers `None` and serves the call through `generate`; it is never served by its other dialect.
- **You hand over one canonical schema.** The leaf converts it with the pure `to_dialect(schema, dialect)`: the open dialects send it as it is; `openai_strict` closes every object, requires every key, and widens a key that was optional to `type | null`. A chain passes the schema on untouched. A keyword outside `cogno_synapse.schema.DIALECT_KEYWORDS` raises `ValueError` before any request — that table is a working hypothesis (the small set a derived reply schema uses), not a measurement of any provider.
- **The OpenAI parameter is built in one place**: `cogno_synapse.openai_backend.response_format_for(schema, name, dialect)`, pure. A capability probe that wants to send exactly what a structured call sends uses it. Which models refuse which dialect is a table that starts EMPTY (`SCHEMA_DIALECTS_REFUSED_BY_MODEL`): a row needs the measurement that justifies it.
- **The receipt says what travelled** (`schema_applied_of`, closed `SCHEMA_RECEIPTS`): a dialect, `"refused"`, or `None` (no schema went). Like `cached_tokens_of`, it is a per-instance last-call value: read it immediately after the `await`, in one event loop.
- **A refused schema does not cost the turn.** A 400 on a structured call is repeated ONCE on the same backend without the schema (receipt `"refused"`, log `event=schema_refused provider=… model=… dialect=… schema_sha=…`), and that instance does not send that schema to that model again for 15 minutes. The failover loop never sees it. A 429, a timeout or a 5xx propagates as always.
- **`system` and `prompt` are the same bytes on both paths.** The schema rides outside the prompt.

`cogno_synapse.schema.BACKEND_SCHEMA_SUPPORT` holds one row per class of this package that implements `generate` — `structured` or `plain: <reason>` — and a test derives the class list by AST, so a new backend without a row fails.

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

## Environment

The library reads these variables (`os.environ`/`os.getenv` under `cogno_synapse/`). An explicit constructor argument always wins over the variable.

| variable | read by | notes |
|---|---|---|
| `OPENAI_API_KEY` | `OpenAIBackend`, `OpenAIEmbedder`, the factories | |
| `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` | the matching backend (and `GeminiEmbedder`), the factories | |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | `BedrockBackend`, `BedrockEmbedder`; the factories check the first | `BedrockBackend` takes no key argument |
| `AWS_DEFAULT_REGION`, `AWS_REGION` | `BedrockBackend` (the first, then the second; no default) · `BedrockEmbedder` (`AWS_REGION`, default `us-east-1`) | |
| `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY` | `create_backend` for the OpenAI-compatible prefixes | |
| `OPENAI_REASONING_EFFORT` | `OpenAIBackend` | unset = the field is not sent |
| `COGNO_OLLAMA_TIMEOUT` | `OllamaBackend`, `OllamaEmbedder` | seconds, default 120; read per construction and only when no `timeout=` is passed — `create_backend` passes 600 and `create_embedder` 120, so through the factories it does not apply; a bad value logs `event=bad_timeout_env` and falls back to 120 |

A cloud key that is empty, `dummy`, or contains `sk-proj-*` counts as missing for the factories, which then raise `MissingAPIKeyError` unless `api_key=` is passed. A backend constructed directly without its key does not raise: it logs a WARNING at construction, and the failure arrives on the first call.

The tests read three more: `OLLAMA_BASE_URL` (default `http://localhost:11434`), `COGNO_TEST_OLLAMA_MODEL` (default `qwen3:8b`) and `COGNO_TEST_OLLAMA_EMBED_MODEL` (default `nomic-embed-text`), all in `tests/integration/test_ollama_live.py`; the cloud integration tests skip unless the provider's key (and SDK) is present.

## Test

```bash
pip install -e ".[dev]"      # unit tests mock every SDK client and HTTP call — no cloud keys needed
pytest tests/unit -q
```

CI (`.github/workflows/ci.yml`) installs `.[dev,llm]` and runs `ruff check cogno_synapse tests examples`, `mypy cogno_synapse` and `pytest tests -q`. That last one collects the integration suite too: it runs against a reachable Ollama and real provider keys, and skips without them.

## Documentation

| file | what it is |
|---|---|
| [`README.md`](README.md) | this page |
| [`docs/HOST_INTEGRATION.md`](docs/HOST_INTEGRATION.md) | wiring the library into a host: the boundary, the factory, failover, schemas, errors, tokens |
| [`LOGGING.md`](LOGGING.md) | every log line the library emits, and the convention behind them |
| [`CHANGELOG.md`](CHANGELOG.md) | release notes |
| [`examples/host_min.py`](examples/host_min.py) | the smallest runnable host (needs a local Ollama) |
