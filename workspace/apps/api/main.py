"""FastAPI management API service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from packages.config.settings import get_settings
from packages.database.database import db, get_db
from packages.providers.factory import provider_factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()

    # Startup
    db.initialize()
    await provider_factory.get_cache_provider()

    # Validate credentials if using real providers
    if settings.providers_mode == "real":
        missing = settings.validate_required_credentials()
        if missing:
            print(f"WARNING: Missing credentials: {', '.join(missing)}")

    yield

    # Shutdown
    await db.close()
    await provider_factory.close_all()


app = FastAPI(
    title="SVKM Voice Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

settings = get_settings()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "api",
        "version": settings.app_version,
        "environment": settings.app_env
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "SVKM NMIMS Voice Assistant API",
        "version": settings.app_version,
        "docs": "/docs"
    }


# Admin endpoints
@app.get("/admin/sources")
async def list_sources(db: AsyncSession = Depends(get_db)):
    """List knowledge sources."""
    return {"sources": []}


@app.post("/admin/sources/ingest")
async def trigger_ingestion(db: AsyncSession = Depends(get_db)):
    """Trigger knowledge ingestion."""
    return {"status": "started", "run_id": "placeholder"}


@app.get("/admin/contacts")
async def list_contacts(db: AsyncSession = Depends(get_db)):
    """List contact routes."""
    from packages.database.models import ContactRoute
    from sqlalchemy import select

    stmt = select(ContactRoute).where(ContactRoute.is_active == True)
    result = await db.execute(stmt)
    contacts = result.scalars().all()

    return {
        "contacts": [
            {
                "id": str(c.id),
                "school": c.school.value,
                "primary_phone": c.primary_phone,
                "fallback_phones": c.fallback_phones
            }
            for c in contacts
        ]
    }


@app.get("/analytics/overview")
async def analytics_overview(db: AsyncSession = Depends(get_db)):
    """Get analytics overview."""
    return {
        "total_calls": 0,
        "completed_calls": 0,
        "transferred_calls": 0,
        "average_duration": 0,
        "language_distribution": {},
        "school_distribution": {}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload
    )
