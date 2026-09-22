"""Seed initial contact routes."""

import asyncio
from uuid import uuid4
from sqlalchemy import select

from packages.database.database import db
from packages.database.models import ContactRoute
from packages.schemas.schemas import School


async def seed_contacts():
    """Seed initial contact routes."""
    db.initialize()

    async with db.session() as session:
        # Check if already seeded
        stmt = select(ContactRoute)
        result = await session.execute(stmt)
        existing = result.scalars().all()

        if existing:
            print("Contact routes already seeded")
            return

        # Seed contacts
        contacts = [
            ContactRoute(
                id=uuid4(),
                school=School.STME,
                primary_phone="9425685966",
                fallback_phones=["9764405069"],
                email="admissions.stme@svkmnmimsgu.ac.in",
                is_active=True
            ),
            ContactRoute(
                id=uuid4(),
                school=School.SPTM,
                primary_phone="9158947999",
                fallback_phones=["8668501496"],
                email="admissions.pharmacy@svkmnmimsgu.ac.in",
                is_active=True
            ),
            ContactRoute(
                id=uuid4(),
                school=School.SC,
                primary_phone="8788701642",
                fallback_phones=[],
                email="admissions.commerce@svkmnmimsgu.ac.in",
                is_active=True
            ),
            ContactRoute(
                id=uuid4(),
                school=School.GENERAL,
                primary_phone="9425685966",
                fallback_phones=[],
                email="admissions@svkmnmimsgu.ac.in",
                is_active=True
            )
        ]

        session.add_all(contacts)
        await session.commit()

        print(f"Seeded {len(contacts)} contact routes")

    await db.close()


if __name__ == "__main__":
    asyncio.run(seed_contacts())
