"""Base repository with common async database operations (raw SQL, asyncpg)."""

from typing import Any, Optional

from app.core.logging import get_logger
from app.repositories.db_pool import get_pool

logger = get_logger(__name__)


class PostgresBaseRepository:
    """Base class for PostgreSQL repositories using raw SQL with asyncpg."""

    async def execute_query(self, query: str, *args) -> list[dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts."""
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def execute_one(self, query: str, *args) -> Optional[dict[str, Any]]:
        """Execute a SELECT query and return a single result."""
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def execute_write(self, query: str, *args) -> str:
        """Execute an INSERT/UPDATE/DELETE and return status string."""
        pool = get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def execute_insert_returning(self, query: str, *args) -> Optional[dict[str, Any]]:
        """Execute INSERT with RETURNING and return the inserted row."""
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
