"""Scoring history CRUD using raw SQL with asyncpg."""
from typing import Any, Optional

from app.core.logging import get_logger
from app.repositories.postgres_base import PostgresBaseRepository

logger = get_logger(__name__)


class ScoringHistoryRepository(PostgresBaseRepository):
    """Repository for scoring_history table (audit trail)."""

    @staticmethod
    def _normalise_record(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Decode JSON fields into Python objects for downstream callers."""
        if record is None:
            return None
        scoring_params = record.get("scoring_params")
        if isinstance(scoring_params, str):
            record["scoring_params"] = json.loads(scoring_params)
        return record

    async def create(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Insert a new scoring history entry and return it."""
        query = """
            INSERT INTO scoring_history (
                match_id, scoring_params, computed_score, computation_time_ms
            )
            VALUES ($1, $2::jsonb, $3, $4)
            RETURNING *
        """
        record = await self.execute_insert_returning(
            query,
            data["match_id"],
            data["scoring_params"],
            data["computed_score"],
            data.get("computation_time_ms"),
        )
        return self._normalise_record(record)

    async def get_by_match_id(self, match_id: str) -> list[dict[str, Any]]:
        """Retrieve all scoring history entries for a match."""
        query = """
            SELECT * FROM scoring_history
            WHERE match_id = $1
            ORDER BY created_at DESC
        """
        records = await self.execute_query(query, match_id)
        return [self._normalise_record(record) for record in records if record is not None]
