# Changelog

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
