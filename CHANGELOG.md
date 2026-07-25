# Changelog

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
