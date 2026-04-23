"""Orchestrates program and scholarship matching evaluations."""

import asyncio
import time
from typing import Any, Optional

from app.config import settings
from app.core.logging import get_logger
from app.models.matching_models import ConfidenceLevel, EntityType
from app.repositories.postgres_attribution_repo import AttributionReportRepository
from app.repositories.postgres_history_repo import ScoringHistoryRepository
from app.repositories.postgres_match_repo import MatchResultRepository
from app.services.embedding_service import EmbeddingService
from app.services.explainability_service import ExplainabilityService
from app.services.llm_service import LLMPipelineService
from app.services.scoring.program_scorer import ProgramScorer
from app.services.scoring.scholarship_scorer import ScholarshipScorer

logger = get_logger(__name__)


class MatchingService:
    """Top-level service that orchestrates the full evaluation pipeline."""

    _BATCH_EVALUATION_CONCURRENCY = 4

    def __init__(self):
        self.match_repo = MatchResultRepository()
        self.attribution_repo = AttributionReportRepository()
        self.history_repo = ScoringHistoryRepository()
        self.embedding_service = EmbeddingService()
        self.explainability_service = ExplainabilityService()
        self.llm_service = LLMPipelineService()

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
            research_alignment = await self._get_research_alignment(user_profile, entity_data)
            total_score, breakdown, metadata = ProgramScorer.compute_score(
                user_profile,
                entity_data,
                research_similarity=research_alignment.get("similarity", 0.0),
                research_alignment_score=research_alignment.get("score"),
                research_alignment_metadata=research_alignment,
            )
        else:
            research_alignment = None
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
            "llm_model_used": (
                research_alignment.get("model")
                if research_alignment and not research_alignment.get("fallback_used", False)
                else None
            ),
            "llm_fallback_used": research_alignment.get("fallback_used", False) if research_alignment else False,
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
                score_metadata=metadata,
            )
            result["attribution_report"] = attribution
            if attribution and attribution.get("llm_model") and not match_result.get("llm_model_used"):
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

    async def evaluate_batch(
        self,
        user_id: str,
        user_profile: dict[str, Any],
        evaluations: list[dict[str, Any]],
        include_attribution: bool = True,
    ) -> dict[str, Any]:
        """Evaluate multiple entities with bounded concurrency while preserving input order."""
        semaphore = asyncio.Semaphore(self._BATCH_EVALUATION_CONCURRENCY)

        async def run_evaluation(index: int, evaluation: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            item_include_attribution = evaluation.get("include_attribution")
            async with semaphore:
                result = await self.evaluate(
                    user_id=user_id,
                    entity_type=str(evaluation["entity_type"]),
                    entity_id=evaluation["entity_id"],
                    user_profile=user_profile,
                    entity_data=evaluation["entity_data"],
                    include_attribution=(
                        item_include_attribution if item_include_attribution is not None else include_attribution
                    ),
                )
            return index, {
                "entity_type": str(evaluation["entity_type"]),
                "entity_id": evaluation["entity_id"],
                **result,
            }

        indexed_results = await asyncio.gather(
            *(run_evaluation(index, evaluation) for index, evaluation in enumerate(evaluations))
        )
        results = [result for _, result in sorted(indexed_results, key=lambda item: item[0])]

        return {
            "count": len(results),
            "results": results,
        }

    async def get_results_by_user(
        self,
        user_id: str,
        entity_type: Optional[str] = None,
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

    async def get_result_by_id(self, match_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single match result by ID."""
        return await self.match_repo.get_by_id(match_id)

    async def _get_research_alignment(
        self,
        user_profile: dict[str, Any],
        program: dict[str, Any],
    ) -> dict[str, Any]:
        """Get research alignment via LLM first, then fall back to pgvector similarity."""
        research_interests = user_profile.get("research_interests", "")
        if not research_interests:
            return {
                "score": 0.0,
                "similarity": 0.0,
                "provider": "none",
                "model": None,
                "fallback_used": False,
                "alignment_summary": "",
            }

        try:
            llm_result = await self.llm_service.assess_research_alignment(
                user_profile=user_profile, entity_data=program
            )
            llm_result.setdefault("similarity", llm_result.get("score", 0.0) / 100.0)
            return llm_result
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("research_alignment_llm_failed", error=str(exc))

        try:
            results = await self.embedding_service.search_similar(
                query_text=research_interests if isinstance(research_interests, str) else " ".join(research_interests),
                top_k=1,
            )
            if results:
                similarity = float(results[0].get("similarity_score", 0.0))
                return {
                    "score": similarity * 100.0,
                    "similarity": similarity,
                    "provider": "pgvector_fallback",
                    "model": None,
                    "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                    "fallback_used": True,
                    "alignment_summary": "Fallback vector similarity used after LLM alignment failed.",
                }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("research_alignment_fallback_failed", error=str(exc))

        return {
            "score": 0.0,
            "similarity": 0.0,
            "provider": "unavailable",
            "model": None,
            "fallback_used": True,
            "alignment_summary": "",
        }

    @staticmethod
    def _determine_confidence(score: float) -> str:
        """Map score to confidence level."""
        if score >= settings.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        if score >= settings.MATCH_SCORE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
