"""Scoring history CRUD using raw SQL with asyncpg."""

import json
from typing import Any

from app.core.logging import get_logger
from app.repositories.postgres_base import PostgresBaseRepository

logger = get_logger(__name__)


class ScoringHistoryRepository(PostgresBaseRepository):
    """Repository for scoring_history table (audit trail)."""

    async def create(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Insert a new scoring history entry and return it."""
        query = """
            INSERT INTO scoring_history (
                match_id, scoring_params, computed_score, computation_time_ms
            )
            VALUES ($1, $2::jsonb, $3, $4)
            RETURNING *
        """
        return await self.execute_insert_returning(
            query,
            data["match_id"],
            json.dumps(data["scoring_params"]),
            data["computed_score"],
            data.get("computation_time_ms"),
        )

    async def get_by_match_id(self, match_id: str) -> list[dict[str, Any]]:
        """Retrieve all scoring history entries for a match."""
        query = """
            SELECT * FROM scoring_history
            WHERE match_id = $1
            ORDER BY created_at DESC
        """
        return await self.execute_query(query, match_id)
