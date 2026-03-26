"""Tests for scholarship match scoring logic."""

from app.services.scoring.scholarship_scorer import ScholarshipScorer


def test_full_eligibility():
    user_profile = {
        "gpa_normalized": 3.8,
        "citizenship": "US",
        "field_of_study": "Computer Science",
        "degree_level": "Masters",
        "activities": ["leadership", "community service"],
        "achievements": ["dean_list"],
        "financial_need_usd": 30000,
    }
    scholarship = {
        "minimum_gpa": 3.5,
        "eligible_citizenships": ["US", "Canada"],
        "eligible_fields_of_study": ["Computer Science", "Data Science"],
        "required_degree_level": "Masters",
        "preferred_criteria": ["leadership", "community service"],
        "award_amount_usd": 25000,
        "acceptance_rate": 0.15,
    }

    total, breakdown, metadata = ScholarshipScorer.compute_score(user_profile, scholarship)

    assert total > 50
    assert breakdown["eligibility"] > 0
    assert metadata["eligibility"]["criteria_met"] == 4


def test_failed_gpa_eligibility():
    user_profile = {"gpa_normalized": 2.8, "citizenship": "US"}
    scholarship = {"minimum_gpa": 3.5, "eligible_citizenships": ["US"]}

    _total, _breakdown, metadata = ScholarshipScorer.compute_score(user_profile, scholarship)

    assert metadata["eligibility"]["details"]["gpa"] == "not_met"
    assert metadata["eligibility"]["details"]["citizenship"] == "met"


def test_no_competition_data():
    user_profile = {"gpa_normalized": 3.5}
    scholarship = {}

    _total, _breakdown, metadata = ScholarshipScorer.compute_score(user_profile, scholarship)

    assert metadata["competition_estimate"]["reason"] == "no_competition_data"


def test_high_competition():
    user_profile = {"gpa_normalized": 3.5}
    scholarship = {"applicant_count": 5000}

    _total, _breakdown, metadata = ScholarshipScorer.compute_score(user_profile, scholarship)

    assert metadata["competition_estimate"]["source"] == "applicant_count_heuristic"


def test_funding_coverage():
    user_profile = {"gpa_normalized": 3.5, "financial_need_usd": 40000}
    scholarship = {"award_amount_usd": 20000}

    _total, _breakdown, metadata = ScholarshipScorer.compute_score(user_profile, scholarship)

    assert metadata["funding_coverage"]["coverage_ratio"] == 0.5
