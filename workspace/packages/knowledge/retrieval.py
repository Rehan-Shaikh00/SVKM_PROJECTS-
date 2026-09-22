"""Simple retrieval service for knowledge base."""

from typing import List, Optional, Any
from dataclasses import dataclass
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from packages.database.models import KBChunk
from packages.schemas.schemas import LanguageCode, PublicationStatus


@dataclass
class RetrievalResult:
    """Retrieval result."""
    chunk_id: UUID
    content: str
    score: float
    metadata: dict


class RetrievalService:
    """Simple hybrid retrieval service."""

    def __init__(self, db_session: AsyncSession, embedding_provider):
        self.db = db_session
        self.embeddings = embedding_provider

    async def search(
        self,
        query: str,
        language: LanguageCode,
        context: Any,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """Hybrid search using pgvector + full-text."""
        # Generate query embedding
        embedding_result = await self.embeddings.embed_text(query)
        query_embedding = embedding_result.embedding

        # Vector similarity search
        stmt = (
            select(KBChunk)
            .where(KBChunk.status == PublicationStatus.PUBLISHED)
            .order_by(KBChunk.embedding.l2_distance(query_embedding))
            .limit(top_k)
        )

        result = await self.db.execute(stmt)
        chunks = result.scalars().all()

        # Convert to retrieval results
        results = [
            RetrievalResult(
                chunk_id=chunk.id,
                content=chunk.content,
                score=0.8,  # Placeholder score
                metadata=chunk.metadata or {}
            )
            for chunk in chunks
        ]

        return results
