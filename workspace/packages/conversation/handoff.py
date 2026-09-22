"""Contact routing and handoff service."""

from typing import Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.database.models import ContactRoute
from packages.schemas.schemas import School


class HandoffService:
    """Handoff and routing service."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get_contact_route(self, school: Optional[School]) -> Optional[ContactRoute]:
        """Get contact route for school."""
        if not school:
            school = School.GENERAL

        stmt = select(ContactRoute).where(
            ContactRoute.school == school,
            ContactRoute.is_active == True
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_handoff_summary(
        self,
        language: str,
        question: str,
        school: Optional[str],
        reason: str,
        citations: list
    ) -> str:
        """Create redacted handoff summary."""
        summary_parts = [
            f"Language: {language}",
            f"Question: {question[:100]}...",
            f"Reason: {reason}"
        ]

        if school:
            summary_parts.append(f"School: {school}")

        return "\n".join(summary_parts)
