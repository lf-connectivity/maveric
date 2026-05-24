"""LLM API client wrapper with multi-provider support.

Thin wrapper around the shared LLM provider at :mod:`radp.agentic.llm.provider`.
Public surface (``LLMClient`` class, ``generate_structured`` and
``generate_text`` methods) is unchanged so existing call sites in
``chains/*.py`` keep working.
"""
from typing import Optional, Type, TypeVar

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from radp.agentic.llm.provider import create_llm_provider
from radp.digital_twin.agentic_mobility.config import Config

T = TypeVar("T", bound=BaseModel)


def _build_provider_config() -> dict:
    """Translate the agentic_mobility ``Config`` class into a provider-agnostic dict."""
    cfg: dict = {
        "provider": Config.LLM_PROVIDER,
        "temperature": 0.0,  # Deterministic outputs for structured generation
    }
    if Config.LLM_PROVIDER == "groq":
        cfg["api_key"] = Config.GROQ_API_KEY
        cfg["model"] = Config.GROQ_MODEL
    elif Config.LLM_PROVIDER == "bedrock":
        cfg["model"] = Config.BEDROCK_MODEL
        cfg["region"] = Config.BEDROCK_REGION
        if Config.AWS_ACCESS_KEY_ID:
            cfg["aws_access_key_id"] = Config.AWS_ACCESS_KEY_ID
            cfg["aws_secret_access_key"] = Config.AWS_SECRET_ACCESS_KEY
    return cfg


class LLMClient:
    """Wrapper for LLM API calls with structured output support."""

    def __init__(self):
        """Initialize LLM client using the shared agentic LLM provider."""
        Config.validate()
        self.provider = create_llm_provider(_build_provider_config())

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def generate_structured(
        self,
        prompt: str,
        output_model: Type[T],
        system_message: Optional[str] = None,
    ) -> T:
        """Generate structured output validated against a Pydantic model.

        Args:
            prompt: User prompt
            output_model: Pydantic model class for output
            system_message: Optional system message

        Returns:
            Instance of ``output_model``
        """
        return self.provider.generate_structured(
            prompt=prompt,
            output_model=output_model,
            system_message=system_message,
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def generate_text(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate free-text output.

        Args:
            prompt: User prompt
            system_message: Optional system message

        Returns:
            Generated text
        """
        if system_message:
            prompt = f"{system_message}\n\n{prompt}"
        return self.provider.generate(prompt)
