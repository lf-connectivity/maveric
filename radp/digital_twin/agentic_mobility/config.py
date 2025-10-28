"""Configuration management for agentic mobility generation."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration class."""

    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")

    # Geocoding Configuration
    GEOCODING_CACHE_ENABLED = os.getenv("GEOCODING_CACHE_ENABLED", "true").lower() == "true"
    GEOCODING_CACHE_TTL = int(os.getenv("GEOCODING_CACHE_TTL", "3600"))

    # Retry Configuration
    MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "2"))
    RETRY_TIMEOUT_SECONDS = int(os.getenv("RETRY_TIMEOUT_SECONDS", "30"))

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file")
