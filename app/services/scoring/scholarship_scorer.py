"""Scholarship match scoring logic (rule-based)."""

from typing import Any

from app.core.logging import get_logger
from app.services.scoring.weights import ScholarshipScoringWeights

logger = get_logger(__name__)


class ScholarshipScorer:
    """Compute scholarship match scores with explainability."""

    @classmethod
    def compute_score(
        cls,
        user_profile: dict[str, Any],
        scholarship: dict[str, Any],
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        """Compute weighted scholarship match score.

        Args:
            user_profile: Student profile data.
            scholarship: Scholarship criteria and details.

        Returns:
            (total_score, score_breakdown, metadata)
        """
        breakdown: dict[str, float] = {}
        metadata: dict[str, Any] = {}

        eligibility_score, eligibility_meta = cls._score_eligibility(user_profile, scholarship)
        breakdown["eligibility"] = eligibility_score * ScholarshipScoringWeights.ELIGIBILITY / 100
        metadata["eligibility"] = eligibility_meta

        preferred_score, preferred_meta = cls._score_preferred_criteria(user_profile, scholarship)
        breakdown["preferred_criteria"] = preferred_score * ScholarshipScoringWeights.PREFERRED_CRITERIA / 100
        metadata["preferred_criteria"] = preferred_meta

        funding_score, funding_meta = cls._score_funding_coverage(user_profile, scholarship)
        breakdown["funding_coverage"] = funding_score * ScholarshipScoringWeights.FUNDING_COVERAGE / 100
        metadata["funding_coverage"] = funding_meta

        competition_score, competition_meta = cls._score_competition_estimate(scholarship)
        breakdown["competition_estimate"] = competition_score * ScholarshipScoringWeights.COMPETITION_ESTIMATE / 100
        metadata["competition_estimate"] = competition_meta

        total_score = sum(breakdown.values())

        logger.info("scholarship_score_computed", total_score=total_score, breakdown=breakdown)

        return total_score, breakdown, metadata

    @staticmethod
    def _score_eligibility(  # pylint: disable=too-many-branches
        user_profile: dict,
        scholarship: dict,
    ) -> tuple[float, dict]:
        """Score hard eligibility criteria (0-100).

        Checks GPA minimum, citizenship, field of study, etc.
        Binary pass/fail per criterion; overall score is percentage of criteria met.
        """
        criteria_met = 0
        criteria_total = 0
        details: dict[str, Any] = {}

        min_gpa = scholarship.get("minimum_gpa")
        if min_gpa is not None:
            criteria_total += 1
            user_gpa = user_profile.get("gpa_normalized", 0.0)
            if user_gpa >= min_gpa:
                criteria_met += 1
                details["gpa"] = "met"
            else:
                details["gpa"] = "not_met"

        eligible_citizenships = set(scholarship.get("eligible_citizenships", []))
        if eligible_citizenships:
            criteria_total += 1
            user_citizenship = user_profile.get("citizenship", "")
            if user_citizenship in eligible_citizenships:
                criteria_met += 1
                details["citizenship"] = "met"
            else:
                details["citizenship"] = "not_met"

        eligible_fields = set(scholarship.get("eligible_fields_of_study", []))
        if eligible_fields:
            criteria_total += 1
            user_field = user_profile.get("field_of_study", "")
            if user_field in eligible_fields:
                criteria_met += 1
                details["field_of_study"] = "met"
            else:
                details["field_of_study"] = "not_met"

        degree_level = scholarship.get("required_degree_level")
        if degree_level:
            criteria_total += 1
            user_degree = user_profile.get("degree_level", "")
            if user_degree == degree_level:
                criteria_met += 1
                details["degree_level"] = "met"
            else:
                details["degree_level"] = "not_met"

        if criteria_total == 0:
            return 80.0, {"reason": "no_hard_criteria_defined"}

        score = (criteria_met / criteria_total) * 100
        return score, {"criteria_met": criteria_met, "criteria_total": criteria_total, "details": details}

    @staticmethod
    def _score_preferred_criteria(user_profile: dict, scholarship: dict) -> tuple[float, dict]:
        """Score soft / preferred criteria (0-100).

        Checks leadership, community service, extracurriculars, etc.
        """
        preferred = scholarship.get("preferred_criteria", [])
        if not preferred:
            return 60.0, {"reason": "no_preferred_criteria"}

        user_activities = set(user_profile.get("activities", []))
        user_achievements = set(user_profile.get("achievements", []))
        user_all = user_activities | user_achievements

        matched = [c for c in preferred if c in user_all]
        match_ratio = len(matched) / len(preferred)
        score = min(100.0, match_ratio * 130)

        return score, {
            "preferred_count": len(preferred),
            "matched_count": len(matched),
            "match_ratio": match_ratio,
            "matched": matched,
        }

    @staticmethod
    def _score_funding_coverage(user_profile: dict, scholarship: dict) -> tuple[float, dict]:
        """Score how well scholarship funding covers student needs (0-100)."""
        award_amount = scholarship.get("award_amount_usd", 0)
        user_need = user_profile.get("financial_need_usd", 0)

        if user_need <= 0:
            return 80.0, {"reason": "no_financial_need_data"}

        coverage_ratio = award_amount / user_need if user_need > 0 else 1.0
        score = min(100.0, coverage_ratio * 100)

        return score, {
            "award_amount": award_amount,
            "user_need": user_need,
            "coverage_ratio": round(coverage_ratio, 2),
        }

    @staticmethod
    def _score_competition_estimate(scholarship: dict) -> tuple[float, dict]:
        """Estimate competition level (0-100, higher = less competitive = better odds).

        Uses applicant pool size and acceptance rate if available.
        """
        acceptance_rate = scholarship.get("acceptance_rate")
        applicant_count = scholarship.get("applicant_count")

        if acceptance_rate is not None:
            score = min(100.0, acceptance_rate * 100)
            return score, {"acceptance_rate": acceptance_rate, "source": "acceptance_rate"}

        if applicant_count is not None:
            if applicant_count < 50:
                score = 80.0
            elif applicant_count < 200:
                score = 60.0
            elif applicant_count < 1000:
                score = 40.0
            else:
                score = 20.0
            return score, {"applicant_count": applicant_count, "source": "applicant_count_heuristic"}

        return 50.0, {"reason": "no_competition_data"}
