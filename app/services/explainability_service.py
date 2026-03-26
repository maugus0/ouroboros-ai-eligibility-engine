"""Generate attribution reports with LLM-assisted reasoning."""

from typing import Any

from app.core.logging import get_logger
from app.repositories.postgres_attribution_repo import AttributionReportRepository
from app.services.llm_service import LLMPipelineService

logger = get_logger(__name__)


class ExplainabilityService:
    """Generates human-readable attribution reports for match scores."""

    def __init__(self):
        self.attribution_repo = AttributionReportRepository()
        self.llm_service = LLMPipelineService()

    async def generate_report(  # pylint: disable=too-many-locals
        self,
        match_id: str,
        user_profile: dict[str, Any],
        entity_data: dict[str, Any],
        score_breakdown: dict[str, float],
        match_score: float,
    ) -> dict[str, Any] | None:
        """Generate and persist an attribution report.

        Uses rule-based analysis for strengths/gaps, and LLM for reasoning narrative.
        """
        strengths = self._extract_strengths(score_breakdown, match_score)
        gaps = self._extract_gaps(score_breakdown, user_profile, entity_data)
        confidence = self._infer_confidence(match_score)

        reasoning = ""
        llm_provider = None
        llm_model = None
        try:
            llm_result = await self.llm_service.generate_reasoning(
                user_profile=user_profile,
                entity_data=entity_data,
                score_breakdown=score_breakdown,
                strengths=strengths,
                gaps=gaps,
            )
            reasoning = llm_result.get("content", "")
            llm_provider = "openai" if not llm_result.get("fallback_used") else "anthropic"
            llm_model = llm_result.get("model", "")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("llm_reasoning_failed", error=str(exc))
            reasoning = self._fallback_reasoning(strengths, gaps, match_score)
            llm_provider = "rule_based"

        recommendations = self._generate_recommendations(gaps)

        report_data = {
            "match_id": match_id,
            "strengths": strengths,
            "gaps": gaps,
            "reasoning": reasoning,
            "confidence": confidence,
            "recommendations": recommendations,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "prompt_version": "attribution_report_v1",
        }

        return await self.attribution_repo.create(report_data)

    async def get_report(self, match_id: str) -> dict[str, Any] | None:
        """Retrieve an attribution report by match ID."""
        return await self.attribution_repo.get_by_match_id(match_id)

    @staticmethod
    def _extract_strengths(breakdown: dict[str, float], total: float) -> list[str]:
        """Identify score components that performed well."""
        strengths = []
        for component, score in breakdown.items():
            if score >= (total * 0.25):
                strengths.append(f"Strong {component.replace('_', ' ')} score ({score:.1f})")
        if total >= 75:
            strengths.append(f"Overall strong match ({total:.1f}/100)")
        return strengths

    @staticmethod
    def _extract_gaps(
        breakdown: dict[str, float],
        user_profile: dict[str, Any],
        entity_data: dict[str, Any],
    ) -> list[str]:
        """Identify weaknesses and missing requirements."""
        gaps = []
        for component, score in breakdown.items():
            if score < 5.0:
                gaps.append(f"Weak {component.replace('_', ' ')} — needs improvement")

        missing_prereqs = entity_data.get("prerequisites", [])
        completed = set(user_profile.get("completed_courses", []))
        for prereq in missing_prereqs:
            if prereq not in completed:
                gaps.append(f"Missing prerequisite: {prereq}")

        return gaps

    @staticmethod
    def _infer_confidence(score: float) -> str:
        if score >= 75:
            return "high"
        if score >= 50:
            return "medium"
        return "low"

    @staticmethod
    def _fallback_reasoning(strengths: list[str], gaps: list[str], score: float) -> str:
        """Produce a simple rule-based reasoning when LLM is unavailable."""
        parts = [f"Match score: {score:.1f}/100."]
        if strengths:
            parts.append("Strengths: " + "; ".join(strengths) + ".")
        if gaps:
            parts.append("Areas for improvement: " + "; ".join(gaps) + ".")
        return " ".join(parts)

    @staticmethod
    def _generate_recommendations(gaps: list[str]) -> list[dict[str, str]]:
        """Convert gaps into actionable recommendations."""
        recs = []
        for gap in gaps:
            if "prerequisite" in gap.lower():
                recs.append({"action": gap.replace("Missing prerequisite: ", "Complete "), "priority": "high"})
            else:
                recs.append({"action": f"Address: {gap}", "priority": "medium"})
        return recs
