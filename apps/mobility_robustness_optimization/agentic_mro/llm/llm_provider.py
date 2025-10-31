"""
LLM Provider Abstraction for Agentic MRO System

This module provides a unified interface for different LLM providers (Groq, Bedrock, etc.)
allowing easy switching between providers without changing agent code.

Usage:
    config = {
        "provider": "groq",  # or "bedrock"
        "model": "llama-3.1-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 2000
    }
    
    llm = create_llm_provider(config)
    response = llm.generate(prompt)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import os
import json


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
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt string
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> Dict:
        """
        Generate JSON response from LLM.
        
        Args:
            prompt: Input prompt string requesting JSON output
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON dictionary
        """
        pass
    
    def validate_response(self, response: str) -> bool:
        """
        Validate LLM response is non-empty.
        
        Args:
            response: LLM response string
            
        Returns:
            True if valid, False otherwise
        """
        return bool(response and response.strip())


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Groq provider.
        
        Required config:
            - api_key: Groq API key (or set GROQ_API_KEY env var)
            - model: Model name (e.g., "llama-3.1-70b-versatile")
        """
        super().__init__(config)
        
        # Import Groq SDK
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "Groq SDK not installed. Install with: pip install groq"
            )
        
        # Get API key from config or environment
        api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key required. Set 'api_key' in config or GROQ_API_KEY env var"
            )
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        
        # Default model if not specified
        if not self.model:
            self.model = "llama-3.1-70b-versatile"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Groq API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override temperature, max_tokens, etc.
            
        Returns:
            Generated text response
        """
        # Override defaults with kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Call Groq API
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract response
        response = chat_completion.choices[0].message.content
        
        if not self.validate_response(response):
            raise ValueError("Empty response from Groq API")
        
        return response
    
    def generate_json(self, prompt: str, **kwargs) -> Dict:
        """
        Generate JSON response using Groq API.
        
        Args:
            prompt: Input prompt requesting JSON output
            **kwargs: Override generation parameters
            
        Returns:
            Parsed JSON dictionary
        """
        # Generate response
        response = self.generate(prompt, **kwargs)
        
        # Try to parse JSON
        try:
            # Handle case where response has markdown code blocks
            if "```json" in response:
                # Extract JSON from markdown
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                # Extract from generic code block
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            # Parse JSON
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from Groq: {e}\nResponse: {response}")


class BedrockProvider(BaseLLMProvider):
    """Amazon Bedrock LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bedrock provider.

        Required config:
            - model: Model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            - region: AWS region (e.g., "us-east-1")

        Optional config (will use environment variables or AWS default credentials if not provided):
            - aws_access_key_id: AWS Access Key ID (falls back to AWS_ACCESS_KEY_ID env var)
            - aws_secret_access_key: AWS Secret Access Key (falls back to AWS_SECRET_ACCESS_KEY env var)

        Credential Priority:
            1. Credentials from config dict
            2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
            3. AWS default credentials (~/.aws/credentials or IAM role)
        """
        super().__init__(config)

        # Import boto3
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 not installed. Install with: pip install boto3"
            )

        # Get region (from config or env var)
        self.region = config.get("region") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        # Initialize Bedrock client
        # Priority order: 1. Config dict, 2. Environment variables, 3. AWS default credentials
        session_kwargs = {}

        # Check config first, then fall back to environment variables
        aws_access_key = config.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = config.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")

        if aws_access_key and aws_secret_key:
            session_kwargs["aws_access_key_id"] = aws_access_key
            session_kwargs["aws_secret_access_key"] = aws_secret_key

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            **session_kwargs
        )

        # Default model if not specified
        if not self.model:
            self.model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Amazon Bedrock API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override temperature, max_tokens, etc.
            
        Returns:
            Generated text response
        """
        # Override defaults with kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Prepare request based on model family
        if "anthropic.claude" in self.model:
            # Claude models use Messages API format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        elif "amazon.titan" in self.model:
            # Titan models use different format
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
        else:
            raise ValueError(f"Unsupported Bedrock model: {self.model}")
        
        # Call Bedrock API
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body)
        )
        
        # Parse response
        response_body = json.loads(response["body"].read())
        
        # Extract text based on model family
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
        """
        Generate JSON response using Bedrock API.
        
        Args:
            prompt: Input prompt requesting JSON output
            **kwargs: Override generation parameters
            
        Returns:
            Parsed JSON dictionary
        """
        # Generate response
        response = self.generate(prompt, **kwargs)
        
        # Try to parse JSON
        try:
            # Handle case where response has markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from Bedrock: {e}\nResponse: {response}")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation (optional)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.
        
        Required config:
            - api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            - model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        """
        super().__init__(config)
        
        # Import OpenAI SDK
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI SDK not installed. Install with: pip install openai"
            )
        
        # Get API key
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set 'api_key' in config or OPENAI_API_KEY env var"
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Default model
        if not self.model:
            self.model = "gpt-4"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = completion.choices[0].message.content
        
        if not self.validate_response(response):
            raise ValueError("Empty response from OpenAI API")
        
        return response
    
    def generate_json(self, prompt: str, **kwargs) -> Dict:
        """Generate JSON response using OpenAI API."""
        response = self.generate(prompt, **kwargs)
        
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response from OpenAI: {e}\nResponse: {response}")


def create_llm_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    """
    Factory function to create LLM provider based on configuration.
    
    Args:
        config: Configuration dictionary with 'provider' key
            {
                "provider": "groq" | "bedrock" | "openai",
                "model": "model-name",
                "temperature": 0.2,
                "max_tokens": 2000,
                ... (provider-specific params)
            }
    
    Returns:
        Initialized LLM provider instance
    
    Raises:
        ValueError: If provider not supported
    
    Example:
        >>> config = {
        ...     "provider": "groq",
        ...     "model": "llama-3.1-70b-versatile",
        ...     "api_key": "gsk_...",
        ...     "temperature": 0.2
        ... }
        >>> llm = create_llm_provider(config)
        >>> response = llm.generate("What is MRO?")
    """
    provider_type = config.get("provider", "").lower()
    
    if provider_type == "groq":
        return GroqProvider(config)
    elif provider_type == "bedrock" or provider_type == "amazon_bedrock":
        return BedrockProvider(config)
    elif provider_type == "openai":
        return OpenAIProvider(config)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider_type}. "
            f"Supported providers: groq, bedrock, openai"
        )


# Configuration examples for different providers
EXAMPLE_CONFIGS = {
    "groq": {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "api_key": "gsk_...",  # or set GROQ_API_KEY env var
        "temperature": 0.2,
        "max_tokens": 2000
    },
    "bedrock_claude": {
        "provider": "bedrock",
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region": "us-east-1",
        "temperature": 0.2,
        "max_tokens": 2000
        # Uses default AWS credentials
    },
    "bedrock_titan": {
        "provider": "bedrock",
        "model": "amazon.titan-text-express-v1",
        "region": "us-east-1",
        "temperature": 0.2,
        "max_tokens": 2000
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "sk-...",  # or set OPENAI_API_KEY env var
        "temperature": 0.2,
        "max_tokens": 2000
    }
}


if __name__ == "__main__":
    """Test LLM provider with example."""
    
    # Example: Using Groq
    print("Testing LLM Provider Abstraction\n")
    print("=" * 60)
    
    # Test configuration (you need to set your API key)
    test_config = {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 500
    }
    
    try:
        # Create provider
        llm = create_llm_provider(test_config)
        print(f"✓ Created {test_config['provider']} provider")
        
        # Test text generation
        prompt = "Explain what Mobility Robustness Optimization (MRO) is in one sentence."
        response = llm.generate(prompt)
        print(f"\n✓ Text Generation Test:")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        # Test JSON generation
        json_prompt = """Generate a JSON object with the following structure:
        {
            "network_type": "LTE",
            "parameters": ["hysteresis", "time_to_trigger"],
            "optimal_range": {"hysteresis": [3, 5], "ttt": [6, 8]}
        }
        Return only valid JSON, no other text."""
        
        json_response = llm.generate_json(json_prompt)
        print(f"\n✓ JSON Generation Test:")
        print(f"Parsed JSON: {json.dumps(json_response, indent=2)}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("Install required package for your provider:")
        print("  - Groq: pip install groq")
        print("  - Bedrock: pip install boto3")
        print("  - OpenAI: pip install openai")
    
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("Set your API key in environment variable or config")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
