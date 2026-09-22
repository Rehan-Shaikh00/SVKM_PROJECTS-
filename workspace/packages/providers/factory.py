"""Provider factory for creating provider instances."""

from packages.providers.interfaces import (
    SpeechProvider, LLMProvider, EmbeddingProvider,
    BlobStorageProvider, CacheProvider
)
from packages.providers.speech import AzureSpeechProvider, FakeSpeechProvider
from packages.providers.llm import ClaudeProvider, FakeLLMProvider
from packages.providers.embeddings import AzureEmbeddingProvider, FakeEmbeddingProvider
from packages.providers.cache import RedisCache, FakeCache
from packages.config.settings import get_settings


class ProviderFactory:
    """Factory for creating provider instances."""

    def __init__(self):
        self.settings = get_settings()
        self._speech = None
        self._llm = None
        self._embeddings = None
        self._cache = None

    def get_speech_provider(self) -> SpeechProvider:
        """Get speech provider instance."""
        if self._speech is None:
            if self.settings.providers_mode == "fake":
                self._speech = FakeSpeechProvider()
            else:
                self._speech = AzureSpeechProvider()
        return self._speech

    def get_llm_provider(self) -> LLMProvider:
        """Get LLM provider instance."""
        if self._llm is None:
            if self.settings.providers_mode == "fake":
                self._llm = FakeLLMProvider()
            else:
                self._llm = ClaudeProvider()
        return self._llm

    def get_embedding_provider(self) -> EmbeddingProvider:
        """Get embedding provider instance."""
        if self._embeddings is None:
            if self.settings.providers_mode == "fake":
                self._embeddings = FakeEmbeddingProvider()
            else:
                self._embeddings = AzureEmbeddingProvider()
        return self._embeddings

    async def get_cache_provider(self) -> CacheProvider:
        """Get cache provider instance."""
        if self._cache is None:
            if self.settings.providers_mode == "fake":
                self._cache = FakeCache()
            else:
                self._cache = RedisCache()
            await self._cache.connect()
        return self._cache

    async def close_all(self):
        """Close all provider connections."""
        if self._cache:
            await self._cache.close()


# Global factory instance
provider_factory = ProviderFactory()
