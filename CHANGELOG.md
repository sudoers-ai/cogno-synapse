# Changelog

## Unreleased

Documentation only — no code, test or packaging change.

- `README.md`: the install line no longer says "httpx only" (the base install also needs
  `cogno-homeo`, and the OpenAI-compatible providers need the `openai` extra); `OllamaBackend`
  is said to be a plain `LLMBackend` (no `chat_with_tools`); new sections for the embedders and
  `create_embedder`, the three last-call readers (`cached_tokens_of`, `system_fingerprint_of`,
  `served_model_of`), what `OpenAIBackend` does with JSON mode, `reasoning_effort` and a cut
  reply, every environment variable the library reads, the CI commands, and an index of every
  document.
- `LOGGING.md`: rewritten in English from the log calls themselves — every line, its logger,
  level and fields. It used to say INFO is only the "call finished" line (the tool-call parser
  logs three more), that the raw response goes to DEBUG (no line logs it), that loggers are
  named by `__name__` (one module does), and it pointed at a file of another codebase.
- `docs/HOST_INTEGRATION.md`: the `kimi` and `grok` prefix aliases, `api_key=` and the
  placeholder rule, which timeout the factories pass (so `$COGNO_OLLAMA_TIMEOUT` does not reach
  them), the cloud embedders, `EmbeddingResponseError`, and the last-call readers.
- This file: the 0.1.3 entry below now lists what landed between 0.1.2 and the 0.1.3 bump.

## 0.1.3 — 2026-10-14

A JSON Schema can travel WITH the request. Inert: nothing in this package, and no
caller that exists, asks for one — every `generate` call is the request it was.

- `StructuredBackend`, a third optional protocol beside `ToolCallingBackend`:
  `schema_dialect(form)` + `generate_structured(system, prompt, *, schema, name,
  form)`. NEW members, not a new argument: `generate(system, prompt)` keeps its
  two-argument shape on every backend, wrapper and double.
- The form (`"open"` | `"strict"`, `SCHEMA_FORMS`) travels per CALL and has no
  default. Form → dialect per backend, never a demotion: OpenAI `open` →
  `openai_json_schema`, `strict` → `openai_strict`; Ollama `open` →
  `ollama_format`, `strict` → `None`; Anthropic, Gemini, Groq, Bedrock and every
  `base_url` endpoint → `None`; a `FallbackBackend` → `"chain"` when any link
  answers that form.
- `cogno_synapse.schema.to_dialect(schema, dialect)` — the pure conversion. The
  caller hands over one canonical schema; the LEAF converts, a chain does not.
  A keyword outside `DIALECT_KEYWORDS` is a `ValueError` before any request.
  That table is a working hypothesis, not a measurement of any provider.
- `cogno_synapse.openai_backend.response_format_for(schema, name, dialect)` builds
  the OpenAI parameter in one place (pure), and `SCHEMA_DIALECTS_REFUSED_BY_MODEL`
  — which models refuse which dialect — starts EMPTY: no probe has been run.
- The receipt: `schema_applied_of(backend)`, read immediately after the `await`
  like `cached_tokens_of` — a dialect, `"refused"`, or `None`.
- A 400 on a structured call is ONE repeat on the same backend without the
  schema (receipt `"refused"`), remembered per `(model, schema digest)` for 15
  minutes; it never reaches the failover loop, so no retry is spent and no
  breaker opens. A 429, a timeout or a 5xx propagates as before.
- `OllamaBackend` sends the schema in the `format` field of THAT call;
  `self.format` is never written.
- `BACKEND_SCHEMA_SUPPORT`: one row per class that implements `generate`
  (`structured` | `plain: <reason>`); a test derives the class list by AST.
- `OpenAIBackend.generate` and `OllamaBackend.generate` now run through a private
  helper each (`_complete`, `_post`); the request each sends is unchanged, pinned
  by literal requests in `tests/unit/test_structured_backend.py`.

Also in this version — landed between 0.1.2 and the 0.1.3 bump, and not listed when the
entry above was written:

- **BYOK key probe** (#21, #25, #26): `probe_api_key(provider, api_key)` and the
  `API_KEY_PROBES` table — a cost-free authenticated GET per provider. Fail-open: only a 401
  or a 403 reads invalid; anything else logs `event=byok_probe_inconclusive` (or
  `byok_probe_error`) and reads valid. A blank key is never trusted.
- **Cached prompt tokens** (#22): `cached_tokens_of(backend)`, the part of `tokens_in` OpenAI
  served from its own cache (`usage.prompt_tokens_details.cached_tokens`), forwarded by
  `FallbackBackend` from the link that ran. 0 means unknown or none.
- **Who served the call** (#28): `system_fingerprint_of(backend)` and `served_model_of(backend)`,
  reported by OpenAI and Groq on both the text and the tool path, `None` when the provider
  says nothing, forwarded by `FallbackBackend`.
- **OpenAI JSON mode** (#17, #18): `response_format={"type": "json_object"}` when the SYSTEM
  prompt contains "json" — never read from `prompt`, which can quote the contact — and never on
  a `base_url` endpoint or the reasoning series.
- **Reasoning models and tools** (#19): `reasoning_effort` (argument or
  `$OPENAI_REASONING_EFFORT`), sent only to the reasoning series and, by default, not at all.
  A tool call refused because reasoning is on is repeated once with `reasoning_effort="none"`,
  remembered per instance (`event=reasoning_tools_conflict`).
- **A cut reply is logged** (#15): `event=truncated_response` when `finish_reason` is `length`
  or `content_filter`.
- **Ollama embeddings are metered** (#23): `OllamaEmbedder` posts to `/api/embed`, which
  reports `prompt_eval_count`; on a 404/405 it falls back once per instance to the legacy
  `/api/embeddings` (no count) and logs `event=embed_endpoint_missing`.
- **Security**: the bracket form of the tool-call rescue fires only when the tag is alone on
  its line, so a bracket echoed in prose is never executed; `GeminiEmbedder` sends its key in
  the `x-goog-api-key` header instead of the URL.
- **CI** (#20, #27): one test step over `tests/`, pinned by
  `tests/unit/test_ci_runs_every_test_file.py`; a concurrency group per workflow and ref.

## 0.1.2 — 2026-08-04

The Ollama call timeout stops assuming a GPU.

- `$COGNO_OLLAMA_TIMEOUT` sets the per-call timeout for `OllamaBackend` and
  `OllamaEmbedder` (default 120 s, unchanged). Read per construction rather than
  at import, so a host or a test can set it after the module is loaded; an
  explicit `timeout=` argument still wins; a non-numeric or non-positive value
  logs and falls back rather than raising or silently removing the bound.

  120 s is right for a GPU box and short for ONE generation on an 8B model on
  CPU. That is not hypothetical: the nightly Ollama canaries in `cogno-anima`
  and `cogno-soma` both went red on `httpx.ReadTimeout` — cogno-anima at 40 of
  41 tests passing — with nothing wrong but the clock. Any host doing CPU-only
  inference hits the same wall.

## 0.1.1 — 2026-08-02

Embeddings gain the cloud. The `Embedder` protocol previously had exactly one
implementation, so any consumer wanting a cloud embedder had nothing to build.

- `OpenAIEmbedder` (`/embeddings`), `GeminiEmbedder` (REST, no extra dependency
  — httpx is already required) and `BedrockEmbedder` (Titan + Cohere, whose
  payload shapes differ, so the model id picks the adapter).
- `create_embedder("provider:model")`, mirroring `create_backend` for the other
  protocol. Deliberately returns ONE embedder, never a fallback chain: failing
  over mid-write mixes embedding spaces in a single column, and cosine across
  spaces is meaningless while still returning a plausible number.
- `dimensions` is honoured, which is what lets a cloud embedder drop into a
  store sized for a local one (a v3 OpenAI model at 768 needs no migration).
- `CachingEmbedder` now proxies the batch path; without it the wrapper hid the
  inner `embed_batch` and every consumer silently fell back to N sequential
  calls — the bulk re-embedding case is exactly the one that lost.
- Embedding providers are a CLOSED set (ollama, openai, gemini, bedrock).
  OpenAI-compatible prefixes are refused at construction: sharing
  `/chat/completions` says nothing about `/embeddings`. Chat routing unchanged.

## 0.1.0 — 2026-07-25

First public release on PyPI.

- Model-transport layer: structurally-typed backend protocols (`LLMBackend`,
  `ToolCallingBackend`, `Embedder`) + concrete implementations.
- Local Ollama backend/embedder; cloud backends (OpenAI, Anthropic, Groq,
  Gemini, Bedrock) as lazy-imported optional extras, plus OpenAI-compatible
  providers (DeepSeek, Kimi, Grok, OpenRouter, Together, Fireworks) via
  `base_url`.
- `create_backend("provider:model")` factory, bounded-LRU `CachingEmbedder`,
  and a resilient `FallbackBackend` chain running on the `cogno-homeo` kernel.
