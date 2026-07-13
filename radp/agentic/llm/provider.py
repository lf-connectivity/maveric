"""Shared LLM Provider Abstraction (Sprint 3.2).

Single source of truth used by both:
  - apps/mobility_robustness_optimization/agentic_mro
  - radp/digital_twin/agentic_mobility

Supported providers
-------------------
  - ``groq``      — Groq SDK
  - ``bedrock``   — Amazon Bedrock (boto3); supports Anthropic Claude and Titan
  - ``openai``    — OpenAI SDK
  - ``local``     — Local model via Ollama (lazy import of langchain-community)

Usage
-----
    >>> config = {"provider": "groq", "model": "llama-3.1-70b-versatile"}
    >>> llm = create_llm_provider(config)
    >>> llm.generate("What is MRO?")           # free text -> str
    >>> llm.generate_json("Return {...}")      # parsed JSON -> dict
    >>> llm.generate_structured(prompt, MyPydanticModel)  # validated -> BaseModel

Provider selection priority for ``create_llm_provider``:
    1. ``config["provider"]``
    2. ``os.environ["LLM_PROVIDER"]``
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
import json
import os

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM provider with configuration.

        Args:
            config: Dictionary containing provider-specific configuration
                - model: Model identifier
                - temperature: Sampling temperature (0.0 - 1.0)
                - max_tokens: Maximum tokens in response
                - Additional provider-specific params
        """
        self.config = config
        self.model = config.get("model")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 2000)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate free-text response."""

    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> Dict:
        """Generate response and parse it as JSON."""

    def generate_structured(
        self,
        prompt: str,
        output_model: Type[T],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> T:
        """Generate output validated against a Pydantic model.

        Implemented on top of :meth:`generate` using LangChain's
        ``PydanticOutputParser`` for format instructions and parsing. This is
        the method used by ``agentic_mobility`` chains.
        """
        from langchain_core.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=output_model)
        format_instructions = parser.get_format_instructions()

        full_prompt = f"{prompt}\n\n{format_instructions}"
        if system_message:
            full_prompt = f"{system_message}\n\n{full_prompt}"

        response = self.generate(full_prompt, **kwargs)
        return parser.parse(response)

    def validate_response(self, response: str) -> bool:
        return bool(response and response.strip())


def _extract_json(response: str) -> str:
    """Strip ``` / ```json fences from an LLM response, if any."""
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        return response[start:end].strip()
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        return response[start:end].strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                "Groq SDK not installed. Install with: pip install groq"
            ) from e

        api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key required. Set 'api_key' in config or GROQ_API_KEY env var"
            )

        self.client = Groq(api_key=api_key)
        if not self.model:
            self.model = "llama-3.1-70b-versatile"

    def generate(self, prompt: str, **kwargs) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        response = chat_completion.choices[0].message.content
        if not self.validate_response(response):
            raise ValueError("Empty response from Groq API")
        return response

    def generate_json(self, prompt: str, **kwargs) -> Dict:
        response = self.generate(prompt, **kwargs)
        try:
            return json.loads(_extract_json(response))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response from Groq: {e}\nResponse: {response}"
            ) from e


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------


class BedrockProvider(BaseLLMProvider):
    """Amazon Bedrock LLM provider implementation.

    Credential priority:
        1. Credentials from ``config`` dict
        2. Environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``)
        3. AWS default credentials (``~/.aws/credentials`` or IAM role)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except ImportError as e:
            raise ImportError(
                "boto3 not installed. Install with: pip install boto3"
            ) from e

        self.region = config.get("region") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        session_kwargs: Dict[str, Any] = {}
        aws_access_key = config.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = config.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
        if aws_access_key and aws_secret_key:
            session_kwargs["aws_access_key_id"] = aws_access_key
            session_kwargs["aws_secret_access_key"] = aws_secret_key

        retry_config = BotocoreConfig(
            retries={"max_attempts": 10, "mode": "adaptive"}
        )

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=retry_config,
            **session_kwargs,
        )

        if not self.model:
            self.model = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        if "anthropic.claude" in self.model:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        elif "amazon.titan" in self.model:
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9,
                },
            }
        else:
            raise ValueError(f"Unsupported Bedrock model: {self.model}")

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
        )
        response_body = json.loads(response["body"].read())

        if "anthropic.claude" in self.model:
            text = response_body["content"][0]["text"]
        elif "amazon.titan" in self.model:
            text = response_body["results"][0]["outputText"]
        else:
            raise ValueError(f"Unknown response format for model: {self.model}")

        if not self.validate_response(text):
            raise ValueError("Empty response from Bedrock API")
        return text

    def generate_json(self, prompt: str, **kwargs) -> Dict:
        response = self.generate(prompt, **kwargs)
        try:
            return json.loads(_extract_json(response))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response from Bedrock: {e}\nResponse: {response}"
            ) from e


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK not installed. Install with: pip install openai"
            ) from e

        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set 'api_key' in config or OPENAI_API_KEY env var"
            )

        self.client = OpenAI(api_key=api_key)
        if not self.model:
            self.model = "gpt-4"

    def generate(self, prompt: str, **kwargs) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        response = completion.choices[0].message.content
        if not self.validate_response(response):
            raise ValueError("Empty response from OpenAI API")
        return response

    def generate_json(self, prompt: str, **kwargs) -> Dict:
        response = self.generate(prompt, **kwargs)
        try:
            return json.loads(_extract_json(response))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response from OpenAI: {e}\nResponse: {response}"
            ) from e


# ---------------------------------------------------------------------------
# Local (Ollama)
# ---------------------------------------------------------------------------


class LocalProvider(BaseLLMProvider):
    """Local LLM provider via Ollama.

    Lazy-imports ``langchain_community.llms.Ollama`` so this class is only
    loaded when the user explicitly selects ``provider="local"``.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from langchain_community.llms import Ollama
        except ImportError as e:
            raise ImportError(
                "Ollama support not installed. Install with: "
                "pip install langchain-community"
            ) from e

        base_url = config.get("base_url") or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        )
        if not self.model:
            self.model = config.get("model") or os.getenv("OLLAMA_MODEL", "llama3.1")

        self.client = Ollama(
            model=self.model,
            base_url=base_url,
            temperature=self.temperature,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.invoke(prompt)
        if not self.validate_response(response):
            raise ValueError("Empty response from local LLM (Ollama)")
        return response

    def generate_json(self, prompt: str, **kwargs) -> Dict:
        response = self.generate(prompt, **kwargs)
        try:
            return json.loads(_extract_json(response))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response from local LLM: {e}\nResponse: {response}"
            ) from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_PROVIDERS = {
    "groq": GroqProvider,
    "bedrock": BedrockProvider,
    "amazon_bedrock": BedrockProvider,
    "openai": OpenAIProvider,
    "local": LocalProvider,
    "ollama": LocalProvider,
}


def create_llm_provider(config: Optional[Dict[str, Any]] = None) -> BaseLLMProvider:
    """Build a provider from a config dict.

    Provider selection priority:
        1. ``config["provider"]``
        2. ``os.environ["LLM_PROVIDER"]``

    Args:
        config: Configuration dict. ``provider`` selects the class; all other
            keys are forwarded to the provider constructor.

    Returns:
        Concrete :class:`BaseLLMProvider` instance.

    Raises:
        ValueError: If ``provider`` is missing or unsupported.
    """
    config = dict(config or {})
    provider_type = (config.get("provider") or os.getenv("LLM_PROVIDER", "")).lower()

    if provider_type not in _PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider_type!r}. "
            f"Supported providers: {sorted(set(_PROVIDERS))}"
        )

    return _PROVIDERS[provider_type](config)


# Configuration examples for documentation
EXAMPLE_CONFIGS = {
    "groq": {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    "bedrock_claude": {
        "provider": "bedrock",
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region": "us-east-1",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    "local": {
        "provider": "local",
        "model": "llama3.1",
        "base_url": "http://localhost:11434",
        "temperature": 0.2,
    },
}
