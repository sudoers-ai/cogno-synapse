# Changelog

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
