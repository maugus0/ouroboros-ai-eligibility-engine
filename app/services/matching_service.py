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
            research_alignment = await self._get_research_alignment(user_id, entity_id, user_profile, entity_data)
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
        attribution: Optional[dict[str, Any]] = None

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

        result["agent_reasoning"] = self._build_agent_reasoning(
            entity_type=entity_type,
            total_score=total_score,
            breakdown=breakdown,
            metadata=metadata,
            confidence=confidence,
            include_attribution=include_attribution,
            attribution=attribution,
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
        user_id: str,
        program_id: str,
        user_profile: dict[str, Any],
        program: dict[str, Any],
    ) -> dict[str, Any]:
        """Get research alignment using vector baseline plus LLM semantic reasoning."""
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

        research_interest_text = (
            research_interests if isinstance(research_interests, str) else " ".join(research_interests)
        ).strip()
        program_research_focus = self._extract_program_research_focus(program)
        vector_similarity = 0.0
        vector_metadata: dict[str, Any] = {
            "vector_similarity": 0.0,
            "vector_available": False,
            "program_research_focus": program_research_focus,
        }

        if program_research_focus:
            try:
                vector_result = await self.embedding_service.compute_pair_similarity(
                    student_profile_id=user_id,
                    student_research_text=research_interest_text,
                    program_id=program_id,
                    program_research_text=program_research_focus,
                )
                vector_similarity = float(vector_result.get("similarity", 0.0))
                vector_metadata = {
                    **vector_result,
                    "vector_similarity": vector_similarity,
                    "vector_available": True,
                    "program_research_focus": program_research_focus,
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("research_alignment_vector_failed", error=str(exc))
        else:
            vector_metadata["reason"] = "no_program_research_focus"

        try:
            llm_result = await self.llm_service.assess_research_alignment(
                user_profile=user_profile, entity_data=program
            )
            llm_score = float(llm_result.get("score", 0.0))
            hybrid_score = self._combine_research_alignment_scores(vector_similarity, llm_score)
            llm_result.setdefault("llm_score", llm_score)
            llm_result["score"] = hybrid_score
            llm_result["similarity"] = vector_similarity
            llm_result["vector_similarity"] = vector_similarity
            llm_result["vector_available"] = vector_metadata.get("vector_available", False)
            llm_result["program_research_focus"] = program_research_focus
            llm_result["baseline_provider"] = "pgvector"
            llm_result["scoring_method"] = "hybrid_vector_llm"
            llm_result["student_embedding_reused"] = vector_metadata.get("student_embedding_reused", False)
            llm_result["program_embedding_reused"] = vector_metadata.get("program_embedding_reused", False)
            return llm_result
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("research_alignment_llm_failed", error=str(exc))

        if vector_metadata.get("vector_available"):
            return {
                "score": vector_similarity * 100.0,
                "similarity": vector_similarity,
                "vector_similarity": vector_similarity,
                "provider": "pgvector_fallback",
                "model": None,
                "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                "fallback_used": True,
                "alignment_summary": "Vector similarity fallback used after LLM alignment failed.",
                "program_research_focus": program_research_focus,
                "student_embedding_reused": vector_metadata.get("student_embedding_reused", False),
                "program_embedding_reused": vector_metadata.get("program_embedding_reused", False),
                "scoring_method": "vector_only",
            }

        return {
            "score": 0.0,
            "similarity": 0.0,
            "provider": "unavailable",
            "model": None,
            "fallback_used": True,
            "alignment_summary": "",
            **vector_metadata,
        }

    @staticmethod
    def _determine_confidence(score: float) -> str:
        """Map score to confidence level."""
        if score >= settings.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        if score >= settings.MATCH_SCORE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @staticmethod
    def _extract_program_research_focus(program: dict[str, Any]) -> str:
        """Build a text representation of program research focus from available fields."""
        candidate_keys = [
            "research_focus",
            "research_focus_text",
            "program_research_focus",
            "research_summary",
            "faculty_research",
            "research_areas",
            "keywords",
        ]

        parts: list[str] = []
        for key in candidate_keys:
            value = program.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, list):
                parts.extend(str(item).strip() for item in value if str(item).strip())

        return " ".join(parts).strip()

    @staticmethod
    def _combine_research_alignment_scores(vector_similarity: float, llm_score: float) -> float:
        """Blend vector baseline with LLM reasoning into a 0-100 research alignment score."""
        vector_score = max(0.0, min(100.0, vector_similarity * 100.0))
        llm_score = max(0.0, min(100.0, llm_score))
        if vector_score <= 0.0:
            return llm_score
        return (vector_score * 0.4) + (llm_score * 0.6)

    @classmethod
    def _build_agent_reasoning(
        cls,
        *,
        entity_type: str,
        total_score: float,
        breakdown: dict[str, float],
        metadata: dict[str, Any],
        confidence: str,
        include_attribution: bool,
        attribution: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build deterministic agent_reasoning metadata for match evaluations."""
        decision_factors: list[str] = [
            f"Computed {entity_type} match score {round(total_score, 2)}/100 with confidence {confidence}.",
            cls._build_top_contributors_factor(breakdown),
        ]

        mandatory_failure_message = cls._get_mandatory_failure_message(metadata)
        if mandatory_failure_message:
            decision_factors.append(f"Mandatory eligibility gate failed: {mandatory_failure_message}")

        if include_attribution:
            if isinstance(attribution, dict):
                provider = attribution.get("llm_provider") or "unknown"
                model = attribution.get("llm_model") or "rule_based"
                decision_factors.append(f"Attribution generated via {provider} ({model}).")
            else:
                decision_factors.append("Attribution was requested but could not be generated.")
        else:
            decision_factors.append("Attribution generation was disabled for this evaluation.")

        next_field = cls._derive_next_action(attribution)

        return {
            "approach": """Compute weighted eligibility scores,
                    persist historical evidence, and optionally generate explainability.""",
            "decision_factors": decision_factors,
            "next_field": next_field,
            "confidence": cls._confidence_to_numeric(confidence),
        }

    @staticmethod
    def _build_top_contributors_factor(breakdown: dict[str, float]) -> str:
        sorted_components = sorted(breakdown.items(), key=lambda item: item[1], reverse=True)
        contributors = [
            f"{component.replace('_', ' ')}={round(score, 2)}" for component, score in sorted_components if score > 0
        ][:3]
        if not contributors:
            return "No positive scoring contributors were identified."
        return "Top scoring contributors: " + ", ".join(contributors) + "."

    @staticmethod
    def _get_mandatory_failure_message(metadata: dict[str, Any]) -> Optional[str]:
        eligibility_meta = metadata.get("eligibility") if isinstance(metadata, dict) else None
        if not isinstance(eligibility_meta, dict):
            return None
        if eligibility_meta.get("mandatory_passed", True):
            return None
        failed = eligibility_meta.get("failed_criteria")
        if not isinstance(failed, list):
            return "One or more mandatory eligibility criteria were not met."
        for item in failed:
            if isinstance(item, dict):
                message = str(item.get("message") or "").strip()
                if message:
                    return message
        return "One or more mandatory eligibility criteria were not met."

    @staticmethod
    def _derive_next_action(attribution: Optional[dict[str, Any]]) -> Optional[str]:
        if not isinstance(attribution, dict):
            return None
        recommendations = attribution.get("recommendations")
        if not isinstance(recommendations, list) or not recommendations:
            return None
        first = recommendations[0]
        if not isinstance(first, dict):
            return None
        action = first.get("action")
        if isinstance(action, str) and action.strip():
            return action.strip()
        return None

    @staticmethod
    def _confidence_to_numeric(confidence: str) -> float:
        confidence_value = confidence.value if hasattr(confidence, "value") else confidence
        mapping = {
            "high": 0.9,
            "medium": 0.75,
            "low": 0.6,
        }
        return mapping.get(str(confidence_value).strip().lower(), 0.6)
