"""Health-check endpoints."""

from fastapi import APIRouter

from app.config import APP_VERSION
from app.core.logging import get_logger
from app.repositories.db_pool import get_pool

router = APIRouter(tags=["Health"])
logger = get_logger(__name__)


@router.get("/")
async def root():
    return {
        "message": "Eligibility Engine",
        "version": APP_VERSION,
        "status": "healthy",
    }


@router.get("/health")
async def health_check():
    database_status = "not_connected"
    try:
        pool = get_pool()
        async with pool.acquire() as connection:
            await connection.execute("SELECT 1")
        database_status = "connected"
    except RuntimeError:
        logger.info("health_database_unavailable", reason="pool_not_initialized")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("health_database_check_failed", error=str(exc))

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "database": database_status,
    }
