"""
Async PostgreSQL connection pool using asyncpg.
Raw SQL queries — no ORM.
"""

import json
from dataclasses import dataclass
from typing import Optional

import asyncpg

from app.core.logging import get_logger

logger = get_logger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def _initialise_connection(connection: asyncpg.Connection) -> None:
    """Register codecs so JSON/JSONB columns round-trip as Python objects."""
    await connection.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )
    await connection.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )


@dataclass(frozen=True)
class DatabasePoolConfig:  # pylint: disable=too-many-instance-attributes
    """Parameters for creating the global asyncpg pool."""

    host: str
    port: int
    database: str
    user: str
    password: str
    min_size: int = 5
    max_size: int = 20
    timeout: int = 30


async def create_pool(config: DatabasePoolConfig) -> asyncpg.Pool:
    """Create and cache a global connection pool."""
    global _pool  # pylint: disable=global-statement
    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password,
        init=_initialise_connection,
        min_size=config.min_size,
        max_size=config.max_size,
        timeout=config.timeout,
    )
    logger.info(
        "database_pool_created",
        host=config.host,
        database=config.database,
        min_size=config.min_size,
        max_size=config.max_size,
    )
    return _pool


def get_pool() -> asyncpg.Pool:
    """Return the global pool. Raises if not initialised."""
    if _pool is None:
        raise RuntimeError("Database pool has not been initialised. Call create_pool() first.")
    return _pool


async def close_pool() -> None:
    """Close the connection pool gracefully."""
    global _pool  # pylint: disable=global-statement
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("database_pool_closed")
