"""Attribution reports CRUD using raw SQL with asyncpg."""

import json
from typing import Any

from app.core.logging import get_logger
from app.repositories.postgres_base import PostgresBaseRepository

logger = get_logger(__name__)


class AttributionReportRepository(PostgresBaseRepository):
    """Repository for attribution_reports table."""

    async def create(self, data: dict[str, Any]) -> dict[str, Any] | None:
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
        return await self.execute_insert_returning(
            query,
            data["match_id"],
            json.dumps(data.get("strengths", [])),
            json.dumps(data.get("gaps", [])),
            data["reasoning"],
            data["confidence"],
            json.dumps(data.get("recommendations", [])),
            data.get("llm_provider"),
            data.get("llm_model"),
            data.get("prompt_version"),
        )

    async def get_by_match_id(self, match_id: str) -> dict[str, Any] | None:
        """Retrieve an attribution report by match ID."""
        query = "SELECT * FROM attribution_reports WHERE match_id = $1"
        return await self.execute_one(query, match_id)

    async def get_by_id(self, report_id: str) -> dict[str, Any] | None:
        """Retrieve an attribution report by its own ID."""
        query = "SELECT * FROM attribution_reports WHERE id = $1"
        return await self.execute_one(query, report_id)
