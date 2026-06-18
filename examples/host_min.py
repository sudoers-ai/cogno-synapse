"""
Minimal host wiring for cogno-synapse: pick backends and call them.

cogno-synapse is the transport the mind (cogno-anima) speaks to models through.
A host instantiates a backend (local Ollama or a cloud provider) and an embedder,
optionally wraps them in a resilient FallbackBackend, and injects them into the
cognitive stages.

Requires a reachable local Ollama (or swap for a cloud backend + API key). Run:
    python examples/host_min.py
"""

from __future__ import annotations

import asyncio

from cogno_synapse import (
    CachingEmbedder,
    FallbackBackend,
    OllamaBackend,
    OllamaEmbedder,
    create_backend,
)

# Optional resilience (circuit breaker / retry) comes from the sibling kernel:
from cogno_homeo import CircuitBreaker, RetryPolicy


async def main() -> None:
    # 1) a single backend — by object, or by "provider:model" string.
    gen = OllamaBackend(model="mistral:latest", temperature=0.0)
    # gen = create_backend("openai:gpt-4o-mini")     # cloud, needs OPENAI_API_KEY
    # gen = create_backend("deepseek:deepseek-chat") # OpenAI-compatible via base_url

    # 2) an embedder, cached (bounded LRU + token accounting).
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text:latest"))

    # 3) optional failover chain with a breaker + retry (host opt-in; with none of
    #    these it behaves like a plain "try each once, fail over" chain).
    resilient = FallbackBackend(
        [gen, create_backend("ollama:llama3.2")],
        breaker=CircuitBreaker(fail_threshold=3),
        policy=RetryPolicy(max_retries=1),
    )

    text, tokens_in, tokens_out = await resilient.generate(
        system="You are concise. Answer in one word.",
        prompt="What is the capital of France?",
    )
    print(f"reply: {text!r}  (tokens in={tokens_in} out={tokens_out})")

    vec = await embedder.embed("hello world")
    print(f"embedding dims: {len(vec)}; cache usage: {embedder.usage}")


if __name__ == "__main__":
    asyncio.run(main())
