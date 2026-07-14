"""Simple caching utilities for geocoding results."""
import time
from typing import Any, Dict, Optional

from radp.digital_twin.agentic_mobility.config import Config


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, ttl: int = None):
        """Initialize cache.

        Args:
            ttl: Time to live in seconds (default from config)
        """
        self.ttl = ttl or Config.GEOCODING_CACHE_TTL
        self._cache: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl:
            # Expired
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any):
        """Set value in cache with current timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = (value, time.time())

    def clear(self):
        """Clear all cached values."""
        self._cache.clear()

    def size(self) -> int:
        """Get number of cached items.

        Returns:
            Number of items in cache
        """
        return len(self._cache)


# Global geocoding cache instance
geocoding_cache = SimpleCache() if Config.GEOCODING_CACHE_ENABLED else None
