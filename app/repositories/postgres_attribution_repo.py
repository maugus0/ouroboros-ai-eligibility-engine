"""Attribution reports CRUD using raw SQL with asyncpg."""

import json
from typing import Any, Optional

from app.core.logging import get_logger
from app.repositories.postgres_base import PostgresBaseRepository

logger = get_logger(__name__)


class AttributionReportRepository(PostgresBaseRepository):
    """Repository for attribution_reports table."""

    @staticmethod
    def _normalise_record(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Decode JSON fields into Python objects for downstream callers."""
        if record is None:
            return None
        for field in ("strengths", "gaps", "recommendations"):
            value = record.get(field)
            if isinstance(value, str):
                record[field] = json.loads(value)
        return record

    async def create(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Insert a new attribution report and return it."""
        query = """
            INSERT INTO attribution_reports (
                match_id, strengths, gaps, reasoning, confidence,
                recommendations, llm_provider, llm_model, prompt_version
            )
            VALUES ($1, $2::jsonb, $3::jsonb, $4, $5, $6::jsonb, $7, $8, $9)
            ON CONFLICT (match_id) DO UPDATE SET
                strengths = EXCLUDED.strengths,
                gaps = EXCLUDED.gaps,
                reasoning = EXCLUDED.reasoning,
                confidence = EXCLUDED.confidence,
                recommendations = EXCLUDED.recommendations,
                llm_provider = EXCLUDED.llm_provider,
                llm_model = EXCLUDED.llm_model,
                prompt_version = EXCLUDED.prompt_version
            RETURNING *
        """
        record = await self.execute_insert_returning(
            query,
            data["match_id"],
            data.get("strengths", []),
            data.get("gaps", []),
            data["reasoning"],
            data["confidence"],
            data.get("recommendations", []),
            data.get("llm_provider"),
            data.get("llm_model"),
            data.get("prompt_version"),
        )
        return self._normalise_record(record)

    async def get_by_match_id(self, match_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an attribution report by match ID."""
        query = "SELECT * FROM attribution_reports WHERE match_id = $1"
        return self._normalise_record(await self.execute_one(query, match_id))

    async def get_by_id(self, report_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an attribution report by its own ID."""
        query = "SELECT * FROM attribution_reports WHERE id = $1"
        return self._normalise_record(await self.execute_one(query, report_id))
