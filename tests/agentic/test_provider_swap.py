"""Verify the shared LLM provider swaps cleanly between backends (Sprint 3.2).

All SDK network calls are mocked, so no API keys are required.
"""
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from radp.agentic.llm.provider import (
    BaseLLMProvider,
    BedrockProvider,
    GroqProvider,
    LocalProvider,
    OpenAIProvider,
    create_llm_provider,
)


class DummyOutput(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_provider({"provider": "nonsense"})


def test_factory_uses_env_var_when_config_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with patch("groq.Groq") as MockGroq:
        MockGroq.return_value = MagicMock()
        provider = create_llm_provider({})
        assert isinstance(provider, GroqProvider)


# ---------------------------------------------------------------------------
# Per-provider construction (all SDK calls mocked)
# ---------------------------------------------------------------------------


def test_groq_provider_constructs_and_generates(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with patch("groq.Groq") as MockGroq:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hello from groq"))]
        )
        MockGroq.return_value = mock_client

        provider = create_llm_provider({"provider": "groq", "model": "test"})
        assert isinstance(provider, GroqProvider)
        assert provider.generate("hi") == "hello from groq"


def test_bedrock_provider_constructs_and_generates(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with patch("boto3.client") as MockBoto:
        mock_client = MagicMock()
        body_stream = MagicMock()
        body_stream.read.return_value = b'{"content": [{"text": "hello from bedrock"}]}'
        mock_client.invoke_model.return_value = {"body": body_stream}
        MockBoto.return_value = mock_client

        provider = create_llm_provider(
            {
                "provider": "bedrock",
                "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "region": "us-east-1",
            }
        )
        assert isinstance(provider, BedrockProvider)
        assert provider.generate("hi") == "hello from bedrock"


def _fake_openai_module():
    """Build a fake ``openai`` module with a stub ``OpenAI`` client."""
    fake_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="hello from openai"))]
    )
    fake_module.OpenAI.return_value = mock_client
    return fake_module


def test_openai_provider_constructs_and_generates(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch.dict("sys.modules", {"openai": _fake_openai_module()}):
        provider = create_llm_provider({"provider": "openai", "model": "gpt-4"})
        assert isinstance(provider, OpenAIProvider)
        assert provider.generate("hi") == "hello from openai"


def test_local_provider_constructs_and_generates():
    """LocalProvider lazy-imports Ollama; we patch the import target."""
    mock_ollama_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = "hello from local"
    mock_ollama_cls.return_value = mock_instance

    with patch.dict(
        "sys.modules",
        {"langchain_community.llms": MagicMock(Ollama=mock_ollama_cls)},
    ):
        provider = create_llm_provider({"provider": "local", "model": "llama3.1"})
        assert isinstance(provider, LocalProvider)
        assert provider.generate("hi") == "hello from local"


# ---------------------------------------------------------------------------
# Provider swap — same code, two backends
# ---------------------------------------------------------------------------


def test_provider_swap_changes_backend(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("groq.Groq"), patch.dict(
        "sys.modules", {"openai": _fake_openai_module()}
    ):
        p1 = create_llm_provider({"provider": "groq", "model": "m"})
        p2 = create_llm_provider({"provider": "openai", "model": "gpt-4"})

    assert isinstance(p1, GroqProvider)
    assert isinstance(p2, OpenAIProvider)
    assert type(p1) is not type(p2)


# ---------------------------------------------------------------------------
# Structured output via PydanticOutputParser
# ---------------------------------------------------------------------------


def test_generate_structured_parses_pydantic_model(monkeypatch):
    """generate_structured is the shared method used by agentic_mobility."""

    class _FakeProvider(BaseLLMProvider):
        def generate(self, prompt: str, **kwargs) -> str:
            return '{"name": "alice", "value": 42}'

        def generate_json(self, prompt: str, **kwargs):
            import json as _json

            return _json.loads(self.generate(prompt, **kwargs))

    provider = _FakeProvider({"model": "fake"})
    result = provider.generate_structured("anything", DummyOutput)
    assert isinstance(result, DummyOutput)
    assert result.name == "alice"
    assert result.value == 42
