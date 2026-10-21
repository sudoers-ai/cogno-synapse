# Logging

This library **emits** log records; the **host configures** them (handlers, format, level,
tenant context). Nothing here attaches a handler, sets a formatter or calls `basicConfig`.

## The convention

1. Every logger lives under `cogno_synapse`, so one call sets the level for the whole library:
   `logging.getLogger("cogno_synapse").setLevel(...)`.
2. A line states a fact as `key=value` pairs, formatted lazily
   (`logger.warning("event=… model=%s", model)`), and most lines open with `event=<name>`.
   Tenant, timestamp and channel are not in the message: a host that wants them stamps them on
   every record itself (a `logging.Filter` reading its own context).
3. Levels:
   - **ERROR** — never. A fatal failure is an exception and propagates; the host decides what to
     log.
   - **WARNING** — a condition the library recovered from or that the caller should know about:
     a missing key at construction, a retryable transport error just before it propagates, a
     refused schema, a cut reply.
   - **INFO** — one line per finished text call (`generate_done`), and one per tool-call rescue
     from text.
   - **DEBUG** — the request trace, which carries the prompt. Development only: it holds user
     content and must stay off in a multi-tenant production deployment.
4. The API key is never logged, at any level.

## The loggers

| logger | module |
|---|---|
| `cogno_synapse.openai`, `cogno_synapse.anthropic`, `cogno_synapse.groq`, `cogno_synapse.gemini`, `cogno_synapse.bedrock` | the five cloud backends |
| `cogno_synapse.ollama` | `OllamaBackend` and `OllamaEmbedder` |
| `cogno_synapse.openai_embedder`, `cogno_synapse.gemini_embedder`, `cogno_synapse.bedrock_embedder` | the cloud embedders |
| `cogno_synapse.key_probe` | `probe_api_key` |
| `cogno_synapse.tool_parsing` | `parse_tool_calls_from_text` |

`cogno_synapse/_obs.py` has no logger of its own: its three helpers write to the logger of the
backend that calls them.

## Every line

| level | event | logger | fields | notes |
|---|---|---|---|---|
| DEBUG | `generate` | each backend | `provider model system_len prompt_len prompt` | the full **prompt** (not the system text); built only when DEBUG is on |
| INFO | `generate_done` | each backend | `provider model latency_ms tokens_in tokens_out` | the text path (`generate`) only; `chat_with_tools` emits no such line |
| WARNING | `generate_retryable` | each backend | `provider model status error` | a 429, 500, 502, 503 or 504 just before it propagates (a `FallbackBackend` may then fail over); `error` is the exception's own text |
| WARNING | `truncated_response` | `cogno_synapse.openai` | `provider model finish_reason` | `finish_reason` was `length` or `content_filter`; the text still travels on |
| WARNING | `reasoning_tools_conflict` | `cogno_synapse.openai` | `model` (prefixed `stage=LLM`) | the provider refused tools with reasoning on; the call is repeated with `reasoning_effort="none"` |
| WARNING | `model_refusal` | `cogno_synapse.openai` | `provider model` | on a structured call, the model declined; never the refusal's text |
| WARNING | `schema_refused` | `cogno_synapse.openai`, `cogno_synapse.ollama` | `provider model dialect schema_sha` | a 400 on a schema; the call is repeated without it. Never the schema, never the provider's error text |
| WARNING | `schema_unconvertible` | `cogno_synapse.openai`, `cogno_synapse.ollama` | `provider model dialect reason` | the schema cannot be said in the dialect; `reason` names the keyword and its path, never a value |
| WARNING | `bad_timeout_env` | `cogno_synapse.ollama` | `var value` (prefixed `stage=synapse`) | `$COGNO_OLLAMA_TIMEOUT` is not an integer; 120 s is used |
| WARNING | `embed_endpoint_missing` | `cogno_synapse.ollama` | `base_url` (prefixed `stage=synapse`) | this Ollama has no `/api/embed`; the legacy endpoint is used and embedding tokens meter as 0. Once per instance |
| WARNING | `byok_probe_error` | `cogno_synapse.key_probe` | `provider error` | the probe could not reach the provider; the key reads valid. `error` is the exception's own text |
| WARNING | `byok_probe_inconclusive` | `cogno_synapse.key_probe` | `provider status` | any status ≥ 400 other than 401/403; the key reads valid |
| WARNING | — | each cloud backend and embedder | — | prose, no `event=`: `<VAR> not set — … calls will fail` (Bedrock: `AWS credentials not fully set — … may fail`), at construction |
| INFO | — | `cogno_synapse.tool_parsing` | — | prose, no `event=`: `parse_tool_calls: <n> via XML tags` / `via inline JSON` / `via bracket tags`, when text yields tool calls |

## What a line can carry that this library did not write

- `generate` (DEBUG) carries the prompt, which is the caller's and may be the contact's.
- `generate_retryable` and `byok_probe_error` print the exception with `%s`. That text comes
  from the SDK or the provider, not from this library, so what it contains is theirs.
- `bad_timeout_env` prints the variable's value (`%r`), which is the operator's own setting.

Everything else is a closed name, a count, a status, a model id or a digest.
