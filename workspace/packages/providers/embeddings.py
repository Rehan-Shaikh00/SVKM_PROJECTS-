"""Azure OpenAI embeddings provider."""

import asyncio
from typing import List
from openai import AsyncAzureOpenAI

from packages.providers.interfaces import EmbeddingProvider, EmbeddingResult
from packages.config.settings import get_settings


class AzureEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI embeddings implementation."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncAzureOpenAI(
            api_key=self.settings.azure_openai.key,
            api_version=self.settings.azure_openai.api_version,
            azure_endpoint=str(self.settings.azure_openai.endpoint)
        )

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for single text."""
        response = await self.client.embeddings.create(
            input=text,
            model=self.settings.azure_openai.embedding_deployment
        )

        return EmbeddingResult(
            embedding=response.data[0].embedding,
            model=response.model,
            usage={"prompt_tokens": response.usage.prompt_tokens}
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []

        response = await self.client.embeddings.create(
            input=texts,
            model=self.settings.azure_openai.embedding_deployment
        )

        return [
            EmbeddingResult(
                embedding=data.embedding,
                model=response.model,
                usage={"prompt_tokens": response.usage.prompt_tokens // len(texts)}
            )
            for data in response.data
        ]

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.settings.azure_openai.embedding_dimensions


class FakeEmbeddingProvider(EmbeddingProvider):
    """Fake embedding provider for testing."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Return fake embedding."""
        await asyncio.sleep(0.05)
        return EmbeddingResult(
            embedding=[0.1] * self.dimensions,
            model="fake-embedding",
            usage={"prompt_tokens": len(text) // 4}
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Return fake embeddings."""
        await asyncio.sleep(0.1)
        return [await self.embed_text(t) for t in texts]

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions
