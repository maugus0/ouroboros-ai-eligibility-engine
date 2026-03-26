"""Orchestrates program and scholarship matching evaluations."""

import time
from typing import Any

from app.config import settings
from app.core.logging import get_logger
from app.models.matching_models import ConfidenceLevel, EntityType
from app.repositories.postgres_attribution_repo import AttributionReportRepository
from app.repositories.postgres_history_repo import ScoringHistoryRepository
from app.repositories.postgres_match_repo import MatchResultRepository
from app.services.embedding_service import EmbeddingService
from app.services.explainability_service import ExplainabilityService
from app.services.scoring.program_scorer import ProgramScorer
from app.services.scoring.scholarship_scorer import ScholarshipScorer

logger = get_logger(__name__)


class MatchingService:
    """Top-level service that orchestrates the full evaluation pipeline."""

    def __init__(self):
        self.match_repo = MatchResultRepository()
        self.attribution_repo = AttributionReportRepository()
        self.history_repo = ScoringHistoryRepository()
        self.embedding_service = EmbeddingService()
        self.explainability_service = ExplainabilityService()

    async def evaluate(  # pylint: disable=too-many-locals
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        user_profile: dict[str, Any],
        entity_data: dict[str, Any],
        include_attribution: bool = True,
    ) -> dict[str, Any]:
        """Run the full evaluation pipeline: score → persist → optionally explain.

        Returns:
            Dict with match_result and optional attribution_report.
        """
        start = time.perf_counter()

        if entity_type == EntityType.PROGRAM:
            research_similarity = await self._get_research_similarity(user_profile, entity_data)
            total_score, breakdown, metadata = ProgramScorer.compute_score(
                user_profile, entity_data, research_similarity
            )
        else:
            total_score, breakdown, metadata = ScholarshipScorer.compute_score(user_profile, entity_data)

        confidence = self._determine_confidence(total_score)
        elapsed_ms = round((time.perf_counter() - start) * 1000)

        match_data = {
            "user_id": user_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "match_score": round(total_score, 2),
            "score_breakdown": breakdown,
            "confidence_level": confidence,
            "llm_model_used": None,
            "llm_fallback_used": False,
            "total_processing_time_ms": elapsed_ms,
        }

        match_result = await self.match_repo.create(match_data)

        if match_result:
            await self.history_repo.create(
                {
                    "match_id": str(match_result["id"]),
                    "scoring_params": {
                        "weights": breakdown,
                        "metadata": metadata,
                        "entity_type": entity_type,
                    },
                    "computed_score": round(total_score, 2),
                    "computation_time_ms": elapsed_ms,
                }
            )

        result: dict[str, Any] = {"match_result": match_result}

        if include_attribution and match_result:
            attribution = await self.explainability_service.generate_report(
                match_id=str(match_result["id"]),
                user_profile=user_profile,
                entity_data=entity_data,
                score_breakdown=breakdown,
                match_score=total_score,
            )
            result["attribution_report"] = attribution
            if attribution and attribution.get("llm_model"):
                await self.match_repo.execute_write(
                    "UPDATE match_results SET llm_model_used = $1 WHERE id = $2",
                    attribution["llm_model"],
                    match_result["id"],
                )

        logger.info(
            "evaluation_complete",
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            score=round(total_score, 2),
            confidence=confidence,
            elapsed_ms=elapsed_ms,
        )

        return result

    async def get_results_by_user(
        self,
        user_id: str,
        entity_type: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Retrieve paginated match results for a user."""
        offset = (page - 1) * page_size
        results = await self.match_repo.get_by_user(user_id, entity_type, page_size, offset)
        total = await self.match_repo.count_by_user(user_id, entity_type)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        return {
            "items": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    async def get_result_by_id(self, match_id: str) -> dict[str, Any] | None:
        """Retrieve a single match result by ID."""
        return await self.match_repo.get_by_id(match_id)

    async def _get_research_similarity(
        self,
        user_profile: dict[str, Any],
        _program: dict[str, Any],
    ) -> float:
        """Get research alignment similarity via pgvector if research interests exist."""
        research_interests = user_profile.get("research_interests", "")
        if not research_interests:
            return 0.0

        try:
            results = await self.embedding_service.search_similar(
                query_text=research_interests if isinstance(research_interests, str) else " ".join(research_interests),
                top_k=1,
            )
            if results:
                return results[0].get("similarity_score", 0.0)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("research_similarity_failed", error=str(exc))

        return 0.0

    @staticmethod
    def _determine_confidence(score: float) -> str:
        """Map score to confidence level."""
        if score >= settings.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        if score >= settings.MATCH_SCORE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
