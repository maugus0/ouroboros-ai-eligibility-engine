"""Generate attribution reports with LLM-assisted reasoning."""

from typing import Any, Optional

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
    ) -> Optional[dict[str, Any]]:
        """Generate and persist an attribution report.

        Uses rule-based analysis for strengths/gaps, and LLM for reasoning narrative.
        """
        strengths = self._extract_strengths(score_breakdown, match_score)
        gaps = self._extract_gaps(score_breakdown, user_profile, entity_data)
        rule_confidence = self._infer_rule_confidence(match_score, score_breakdown)
        confidence = rule_confidence

        reasoning = ""
        llm_provider = None
        llm_model = None
        try:
            llm_result = await self.llm_service.generate_attribution(
                match_id=match_id,
                user_profile=user_profile,
                entity_data=entity_data,
                score_breakdown=score_breakdown,
                match_score=match_score,
            )
            reasoning = llm_result.get("reasoning", "")
            llm_provider = llm_result.get("provider", "openai")
            llm_model = llm_result.get("model", "")
            confidence = self._combine_confidence(rule_confidence, llm_result.get("confidence"))
            strengths = self._merge_unique_items(strengths, llm_result.get("strengths", []))
            gaps = self._merge_unique_items(gaps, llm_result.get("gaps", []))
            recommendations = llm_result.get("recommendations") or self._generate_recommendations(gaps)
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

    async def get_report(self, match_id: str) -> Optional[dict[str, Any]]:
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
    def _infer_rule_confidence(score: float, breakdown: dict[str, float]) -> str:
        """Infer confidence from rule-based score consistency and data coverage."""
        active_components = sum(1 for component_score in breakdown.values() if component_score > 0)
        zero_components = sum(1 for component_score in breakdown.values() if component_score <= 0)

        if score >= 75 and active_components >= 4 and zero_components <= 1:
            return "high"
        if score >= 50 and active_components >= 3:
            return "medium"
        return "low"

    @staticmethod
    def _combine_confidence(rule_confidence: str, llm_confidence: Any) -> str:
        """Combine deterministic and LLM confidence signals conservatively."""
        ranking = {"low": 0, "medium": 1, "high": 2}
        llm_confidence_str = str(llm_confidence or rule_confidence).lower()
        llm_confidence_str = llm_confidence_str if llm_confidence_str in ranking else rule_confidence
        final_rank = min(ranking[rule_confidence], ranking[llm_confidence_str])
        return {value: key for key, value in ranking.items()}[final_rank]

    @staticmethod
    def _merge_unique_items(primary: list[str], secondary: list[str]) -> list[str]:
        """Preserve order while merging rule-based and LLM-derived insights."""
        seen: set[str] = set()
        merged: list[str] = []
        for item in [*primary, *secondary]:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
        return merged

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
