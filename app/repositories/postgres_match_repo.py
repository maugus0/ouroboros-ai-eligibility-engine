"""Match results CRUD using raw SQL with asyncpg."""

import json
from typing import Any, Optional

from app.core.logging import get_logger
from app.repositories.postgres_base import PostgresBaseRepository

logger = get_logger(__name__)


class MatchResultRepository(PostgresBaseRepository):
    """Repository for match_results table."""

    @staticmethod
    def _normalise_record(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Decode JSON fields into Python objects for downstream callers."""
        if record is None:
            return None
        score_breakdown = record.get("score_breakdown")
        if isinstance(score_breakdown, str):
            record["score_breakdown"] = json.loads(score_breakdown)
        return record

    async def create(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Insert a new match result and return it."""
        query = """
            INSERT INTO match_results (
                user_id, entity_type, entity_id, match_score,
                score_breakdown, confidence_level, llm_model_used,
                llm_fallback_used, total_processing_time_ms
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9)
            ON CONFLICT (user_id, entity_type, entity_id) DO UPDATE SET
                match_score = EXCLUDED.match_score,
                score_breakdown = EXCLUDED.score_breakdown,
                confidence_level = EXCLUDED.confidence_level,
                llm_model_used = EXCLUDED.llm_model_used,
                llm_fallback_used = EXCLUDED.llm_fallback_used,
                total_processing_time_ms = EXCLUDED.total_processing_time_ms
            RETURNING *
        """
        record = await self.execute_insert_returning(
            query,
            data["user_id"],
            data["entity_type"],
            data["entity_id"],
            data["match_score"],
            data["score_breakdown"],
            data.get("confidence_level"),
            data.get("llm_model_used"),
            data.get("llm_fallback_used", False),
            data.get("total_processing_time_ms"),
        )
        return self._normalise_record(record)

    async def get_by_id(self, match_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a match result by its ID."""
        query = "SELECT * FROM match_results WHERE id = $1"
        return self._normalise_record(await self.execute_one(query, match_id))

    async def get_by_user(
        self,
        user_id: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve match results for a user, optionally filtered by entity type."""
        if entity_type:
            query = """
                SELECT * FROM match_results
                WHERE user_id = $1 AND entity_type = $2
                ORDER BY match_score DESC
                LIMIT $3 OFFSET $4
            """
            records = await self.execute_query(query, user_id, entity_type, limit, offset)
            return [self._normalise_record(record) for record in records if record is not None]

        query = """
            SELECT * FROM match_results
            WHERE user_id = $1
            ORDER BY match_score DESC
            LIMIT $2 OFFSET $3
        """
        records = await self.execute_query(query, user_id, limit, offset)
        return [self._normalise_record(record) for record in records if record is not None]

    async def count_by_user(self, user_id: str, entity_type: Optional[str] = None) -> int:
        """Count match results for a user."""
        if entity_type:
            query = "SELECT COUNT(*) AS cnt FROM match_results WHERE user_id = $1 AND entity_type = $2"
            row = await self.execute_one(query, user_id, entity_type)
        else:
            query = "SELECT COUNT(*) AS cnt FROM match_results WHERE user_id = $1"
            row = await self.execute_one(query, user_id)
        return row["cnt"] if row else 0

    async def delete(self, match_id: str) -> str:
        """Delete a match result by ID."""
        query = "DELETE FROM match_results WHERE id = $1"
        return await self.execute_write(query, match_id)
