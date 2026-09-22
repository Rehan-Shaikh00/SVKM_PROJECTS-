"""Worker service for background jobs."""

import asyncio
import logging

from packages.config.settings import get_settings
from packages.database.database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main worker loop."""
    settings = get_settings()
    logger.info(f"Starting worker service in {settings.app_env} mode")

    db.initialize()

    try:
        while True:
            # Placeholder for background jobs
            logger.info("Worker heartbeat")
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Worker shutting down")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
