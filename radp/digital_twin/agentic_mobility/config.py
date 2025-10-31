"""Configuration management for agentic mobility generation."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration class."""

    # LLM Provider Selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "bedrock"

    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    # Bedrock Configuration
    BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")  # Optional
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")  # Optional

    # Geocoding Configuration
    GEOCODING_CACHE_ENABLED = os.getenv("GEOCODING_CACHE_ENABLED", "true").lower() == "true"
    GEOCODING_CACHE_TTL = int(os.getenv("GEOCODING_CACHE_TTL", "3600"))

    # Retry Configuration
    MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "2"))
    RETRY_TIMEOUT_SECONDS = int(os.getenv("RETRY_TIMEOUT_SECONDS", "30"))

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if cls.LLM_PROVIDER == "groq":
            if not cls.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in .env file")
        elif cls.LLM_PROVIDER == "bedrock":
            # Bedrock uses AWS credentials - boto3 will handle validation
            pass
        else:
            print(cls.LLM_PROVIDER)
            raise ValueError(f"Unsupported LLM_PROVIDER: {cls.LLM_PROVIDER}. Use 'groq' or 'bedrock'")
