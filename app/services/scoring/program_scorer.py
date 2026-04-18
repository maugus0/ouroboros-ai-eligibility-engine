"""Program match scoring logic (rule-based + vector-assisted)."""

from typing import Any, Optional

from app.core.logging import get_logger
from app.services.scoring.weights import ProgramScoringWeights

logger = get_logger(__name__)


class ProgramScorer:
    """Compute program match scores with explainability."""

    @classmethod
    def compute_score(  # pylint: disable=too-many-locals
        cls,
        user_profile: dict[str, Any],
        program: dict[str, Any],
        research_similarity: float = 0.0,
        research_alignment_score: Optional[float] = None,
        research_alignment_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        """Compute weighted program match score.

        Args:
            user_profile: Student profile data.
            program: Program requirements and details.
            research_similarity: Cosine similarity from pgvector (0.0-1.0).
            research_alignment_score: Optional explicit LLM-derived score (0-100).
            research_alignment_metadata: Optional structured metadata for explainability/audit.

        Returns:
            (total_score, score_breakdown, metadata)
        """
        breakdown: dict[str, float] = {}
        metadata: dict[str, Any] = {}

        gpa_score, gpa_meta = cls._score_gpa(user_profile, program)
        breakdown["gpa"] = gpa_score * ProgramScoringWeights.GPA / 100
        metadata["gpa"] = gpa_meta

        relevance_score, relevance_meta = cls._score_relevance(user_profile, program)
        breakdown["relevance"] = relevance_score * ProgramScoringWeights.RELEVANCE / 100
        metadata["relevance"] = relevance_meta

        prereq_score, prereq_meta = cls._score_prerequisites(user_profile, program)
        breakdown["prerequisites"] = prereq_score * ProgramScoringWeights.PREREQUISITES / 100
        metadata["prerequisites"] = prereq_meta

        research_score = research_similarity * 100 if research_alignment_score is None else research_alignment_score
        research_score = max(0.0, min(100.0, research_score))
        breakdown["research_alignment"] = research_score * ProgramScoringWeights.RESEARCH_ALIGNMENT / 100
        metadata["research_alignment"] = {
            "similarity": research_similarity,
            "score": research_score,
            **(research_alignment_metadata or {}),
        }

        practical_score, practical_meta = cls._score_practical_factors(user_profile, program)
        breakdown["practical_factors"] = practical_score * ProgramScoringWeights.PRACTICAL_FACTORS / 100
        metadata["practical_factors"] = practical_meta

        total_score = sum(breakdown.values())

        logger.info("program_score_computed", total_score=total_score, breakdown=breakdown)

        return total_score, breakdown, metadata

    @staticmethod
    def _score_gpa(user_profile: dict, program: dict) -> tuple[float, dict]:
        """Score GPA match (0-100)."""
        user_gpa = user_profile.get("gpa_normalized", 0.0)
        required_gpa = program.get("minimum_gpa", 3.0)

        if user_gpa >= required_gpa:
            score = 100.0
            status = "meets_requirement"
        else:
            score = min(100.0, (user_gpa / required_gpa) * 100) if required_gpa > 0 else 0.0
            status = "below_requirement"

        return score, {"user_gpa": user_gpa, "required_gpa": required_gpa, "status": status}

    @staticmethod
    def _score_relevance(user_profile: dict, program: dict) -> tuple[float, dict]:
        """Score field relevance via keyword overlap (0-100)."""
        user_skills = set(user_profile.get("technical_skills", []))
        program_keywords = set(program.get("keywords", []))

        if not program_keywords:
            return 50.0, {"reason": "no_program_keywords"}

        overlap = user_skills.intersection(program_keywords)
        overlap_ratio = len(overlap) / len(program_keywords)
        score = min(100.0, overlap_ratio * 150)

        return score, {
            "user_skills_count": len(user_skills),
            "program_keywords_count": len(program_keywords),
            "overlap_count": len(overlap),
            "overlap_ratio": overlap_ratio,
            "matched_keywords": sorted(overlap),
        }

    @staticmethod
    def _score_prerequisites(user_profile: dict, program: dict) -> tuple[float, dict]:
        """Score prerequisite completion (0-100)."""
        user_courses = set(user_profile.get("completed_courses", []))
        required_courses = set(program.get("prerequisites", []))

        if not required_courses:
            return 100.0, {"reason": "no_prerequisites"}

        completed = user_courses.intersection(required_courses)
        completion_ratio = len(completed) / len(required_courses)
        score = completion_ratio * 100

        return score, {
            "required_count": len(required_courses),
            "completed_count": len(completed),
            "completion_ratio": completion_ratio,
            "missing_prerequisites": sorted(required_courses - completed),
        }

    @staticmethod
    def _score_practical_factors(user_profile: dict, program: dict) -> tuple[float, dict]:
        """Score practical considerations (0-100)."""
        score = 70.0
        factors = []

        preferred_locations = user_profile.get("preferred_locations", [])
        program_location = program.get("location", "")
        if preferred_locations and program_location in preferred_locations:
            score += 15.0
            factors.append("preferred_location_match")

        user_budget = user_profile.get("budget_usd", float("inf"))
        program_tuition = program.get("tuition_usd", 0)
        if program_tuition <= user_budget:
            score += 15.0
            factors.append("within_budget")

        score = min(100.0, score)

        return score, {
            "factors": factors,
            "location_match": program_location in preferred_locations if preferred_locations else None,
            "budget_match": program_tuition <= user_budget,
        }
