"""
cogno-synapse — the model-transport layer of the Cogno stack.

Named for the *synapse*: the junction that carries the signal between neurons.
Where cogno-anima is the mind and cogno-engram is the memory, cogno-synapse is
the channel the mind speaks to language/embedding models through — a set of
structurally-typed backend protocols plus concrete implementations (Ollama +
cloud providers), a single-backend factory, a resilient fallback chain, and the
text-fallback tool-call parser.

The protocols are the light, stable contract everyone shares; the concrete cloud
backends are optional extras (lazy-imported SDKs). Resilience (breaker/retry/
metrics) is delegated to cogno-homeo.
"""

from cogno_synapse.base import LLMBackend, ToolCallingBackend, Embedder
from cogno_synapse.ollama import OllamaBackend, OllamaEmbedder
from cogno_synapse.cache import CachingEmbedder, EmbeddingUsage
from cogno_synapse.tool_parsing import parse_tool_calls_from_text
from cogno_synapse.openai_backend import OpenAIBackend
from cogno_synapse.anthropic_backend import AnthropicBackend
from cogno_synapse.groq_backend import GroqBackend
from cogno_synapse.gemini_backend import GeminiBackend
from cogno_synapse.bedrock_backend import BedrockBackend
from cogno_synapse.fallback import FallbackBackend
from cogno_synapse.factory import create_backend, parse_model_string
from cogno_synapse.errors import SynapseError, MissingAPIKeyError, InvalidAPIKeyError

__all__ = [
    "LLMBackend",
    "ToolCallingBackend",
    "Embedder",
    "OllamaBackend",
    "OllamaEmbedder",
    "CachingEmbedder",
    "EmbeddingUsage",
    "parse_tool_calls_from_text",
    "OpenAIBackend",
    "AnthropicBackend",
    "GroqBackend",
    "GeminiBackend",
    "BedrockBackend",
    "FallbackBackend",
    "create_backend",
    "parse_model_string",
    "SynapseError",
    "MissingAPIKeyError",
    "InvalidAPIKeyError",
]
