"""Redis cache provider implementation."""

import json
import pickle
from typing import Any, Optional
import redis.asyncio as aioredis

from packages.providers.interfaces import CacheProvider
from packages.config.settings import get_settings


class RedisCache(CacheProvider):
    """Redis cache implementation."""

    def __init__(self):
        self.settings = get_settings()
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        """Connect to Redis."""
        self.redis = await aioredis.from_url(
            str(self.settings.redis.url),
            max_connections=self.settings.redis.max_connections,
            socket_timeout=self.settings.redis.socket_timeout,
            socket_connect_timeout=self.settings.redis.socket_connect_timeout,
            decode_responses=False
        )

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = await self.redis.get(key)
        if value is None:
            return None
        try:
            return pickle.loads(value)
        except:
            return value.decode() if isinstance(value, bytes) else value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache with optional TTL."""
        serialized = pickle.dumps(value)
        if ttl:
            await self.redis.setex(key, ttl, serialized)
        else:
            await self.redis.set(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        await self.redis.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(await self.redis.exists(key))

    async def expire(self, key: str, ttl: int) -> None:
        """Set expiration on key."""
        await self.redis.expire(key, ttl)


class FakeCache(CacheProvider):
    """In-memory fake cache for testing."""

    def __init__(self):
        self.data = {}

    async def connect(self):
        """No-op for fake cache."""
        pass

    async def close(self):
        """No-op for fake cache."""
        pass

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory."""
        return self.data.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in memory (ignores TTL)."""
        self.data[key] = value

    async def delete(self, key: str) -> None:
        """Delete key from memory."""
        self.data.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data

    async def expire(self, key: str, ttl: int) -> None:
        """No-op for fake cache."""
        pass
