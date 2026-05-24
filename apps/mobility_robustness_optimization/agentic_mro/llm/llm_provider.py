"""Re-export shim for the shared LLM provider (Sprint 3.2).

The canonical implementation lives at :mod:`radp.agentic.llm.provider`. This
module re-exports the public API so legacy imports
``from llm.llm_provider import create_llm_provider`` in
``apps/.../agentic_mro/nodes/*.py`` continue to work unchanged.
"""
from radp.agentic.llm.provider import (  # noqa: F401
    EXAMPLE_CONFIGS,
    BaseLLMProvider,
    BedrockProvider,
    GroqProvider,
    LocalProvider,
    OpenAIProvider,
    create_llm_provider,
)

__all__ = [
    "BaseLLMProvider",
    "BedrockProvider",
    "EXAMPLE_CONFIGS",
    "GroqProvider",
    "LocalProvider",
    "OpenAIProvider",
    "create_llm_provider",
]
