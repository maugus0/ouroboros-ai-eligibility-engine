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

        if not eligibility_meta.get("mandatory_passed", True):
            breakdown["preferred_criteria"] = 0.0
            breakdown["funding_coverage"] = 0.0
            breakdown["competition_estimate"] = 0.0
            metadata["preferred_criteria"] = {"score": 0.0, "skipped": True, "reason": "mandatory_eligibility_failed"}
            metadata["funding_coverage"] = {"score": 0.0, "skipped": True, "reason": "mandatory_eligibility_failed"}
            metadata["competition_estimate"] = {"score": 0.0, "skipped": True, "reason": "mandatory_eligibility_failed"}
            metadata["composite"] = {
                "mandatory_eligibility_passed": False,
                "reason": "Mandatory eligibility failed; scholarship match score forced to 0.",
            }
            logger.info("scholarship_score_blocked_by_mandatory_eligibility", metadata=eligibility_meta)
            return 0.0, breakdown, metadata

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
        metadata["composite"] = {
            "mandatory_eligibility_passed": True,
            "total_score": total_score,
        }

        logger.info("scholarship_score_computed", total_score=total_score, breakdown=breakdown)

        return total_score, breakdown, metadata

    @staticmethod
    def _score_eligibility(  # pylint: disable=too-many-branches
        user_profile: dict,
        scholarship: dict,
    ) -> tuple[float, dict]:
        """Evaluate hard eligibility criteria as a mandatory pass/fail gate.

        If any mandatory criterion is not met, the scholarship score is forced to 0.
        """
        criteria_met = 0
        criteria_total = 0
        details: dict[str, Any] = {}
        failed_criteria: list[dict[str, str]] = []

        min_gpa = scholarship.get("minimum_gpa")
        if min_gpa is not None:
            criteria_total += 1
            user_gpa = user_profile.get("gpa_normalized", 0.0)
            if user_gpa >= min_gpa:
                criteria_met += 1
                details["gpa"] = "met"
            else:
                details["gpa"] = "not_met"
                failed_criteria.append(
                    {
                        "criterion": "gpa",
                        "message": f"GPA {user_gpa} is below the minimum requirement of {min_gpa}.",
                    }
                )

        eligible_citizenships = {str(value).strip().lower() for value in scholarship.get("eligible_citizenships", [])}
        if eligible_citizenships:
            criteria_total += 1
            user_citizenship = str(user_profile.get("citizenship", "")).strip().lower()
            if user_citizenship in eligible_citizenships:
                criteria_met += 1
                details["citizenship"] = "met"
            else:
                details["citizenship"] = "not_met"
                failed_criteria.append(
                    {
                        "criterion": "citizenship",
                        "message": "Citizenship does not match the scholarship's eligible countries.",
                    }
                )

        eligible_fields = {str(value).strip().lower() for value in scholarship.get("eligible_fields_of_study", [])}
        if eligible_fields:
            criteria_total += 1
            user_field = str(user_profile.get("field_of_study", "")).strip().lower()
            if user_field in eligible_fields:
                criteria_met += 1
                details["field_of_study"] = "met"
            else:
                details["field_of_study"] = "not_met"
                failed_criteria.append(
                    {
                        "criterion": "field_of_study",
                        "message": "Field of study is outside the scholarship's eligible disciplines.",
                    }
                )

        degree_level = scholarship.get("required_degree_level")
        if degree_level:
            criteria_total += 1
            user_degree = str(user_profile.get("degree_level", "")).strip().lower()
            required_degree = str(degree_level).strip().lower()
            if user_degree == required_degree:
                criteria_met += 1
                details["degree_level"] = "met"
            else:
                details["degree_level"] = "not_met"
                failed_criteria.append(
                    {
                        "criterion": "degree_level",
                        "message": f"Degree level must be {degree_level}.",
                    }
                )

        if criteria_total == 0:
            return 100.0, {"reason": "no_hard_criteria_defined", "mandatory_passed": True, "failed_criteria": []}

        mandatory_passed = len(failed_criteria) == 0
        score = 100.0 if mandatory_passed else 0.0
        return score, {
            "criteria_met": criteria_met,
            "criteria_total": criteria_total,
            "details": details,
            "mandatory_passed": mandatory_passed,
            "failed_criteria": failed_criteria,
        }

    @staticmethod
    def _score_preferred_criteria(user_profile: dict, scholarship: dict) -> tuple[float, dict]:
        """Score soft / preferred criteria (0-100).

        Checks leadership, community service, extracurriculars, etc.
        """
        preferred = scholarship.get("preferred_criteria", [])
        if not preferred:
            return 60.0, {"reason": "no_preferred_criteria"}

        user_all = {
            str(value).strip().lower()
            for value in [
                *user_profile.get("activities", []),
                *user_profile.get("achievements", []),
                *user_profile.get("strengths", []),
                *user_profile.get("tags", []),
            ]
        }

        weighted_items: list[dict[str, Any]] = []
        for item in preferred:
            if isinstance(item, dict):
                label = str(item.get("name") or item.get("criterion") or item.get("key") or "").strip()
                if not label:
                    continue
                aliases = item.get("aliases") or item.get("values") or [label]
                weight = float(item.get("weight", 1.0))
            else:
                label = str(item).strip()
                aliases = [label]
                weight = 1.0

            weighted_items.append(
                {
                    "label": label,
                    "aliases": [str(alias).strip().lower() for alias in aliases if str(alias).strip()],
                    "weight": max(weight, 0.0),
                }
            )

        if not weighted_items:
            return 60.0, {"reason": "no_valid_preferred_criteria"}

        total_weight = sum(item["weight"] for item in weighted_items) or 1.0
        matched: list[str] = []
        unmatched: list[str] = []
        matched_weight = 0.0
        for item in weighted_items:
            if any(alias in user_all for alias in item["aliases"]):
                matched.append(item["label"])
                matched_weight += item["weight"]
            else:
                unmatched.append(item["label"])

        match_ratio = matched_weight / total_weight
        score = min(100.0, match_ratio * 130)

        return score, {
            "preferred_count": len(weighted_items),
            "matched_count": len(matched),
            "match_ratio": match_ratio,
            "matched": matched,
            "unmatched": unmatched,
            "matched_weight": matched_weight,
            "total_weight": total_weight,
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
            return score, {
                "acceptance_rate": acceptance_rate,
                "source": "acceptance_rate",
                "competition_level": ScholarshipScorer._competition_level_from_score(score),
            }

        if applicant_count is not None:
            if applicant_count < 50:
                score = 80.0
            elif applicant_count < 200:
                score = 60.0
            elif applicant_count < 1000:
                score = 40.0
            else:
                score = 20.0
            return score, {
                "applicant_count": applicant_count,
                "source": "applicant_count_heuristic",
                "competition_level": ScholarshipScorer._competition_level_from_score(score),
            }

        return 50.0, {"reason": "no_competition_data", "competition_level": "unknown"}

    @staticmethod
    def _competition_level_from_score(score: float) -> str:
        """Translate win-likelihood score into a competition label."""
        if score >= 75:
            return "low"
        if score >= 50:
            return "moderate"
        if score >= 25:
            return "high"
        return "very_high"
