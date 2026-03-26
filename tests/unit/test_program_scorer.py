"""Tests for program match scoring logic."""

from app.services.scoring.program_scorer import ProgramScorer


def test_perfect_match():
    user_profile = {
        "gpa_normalized": 3.9,
        "technical_skills": ["Python", "Machine Learning", "NLP"],
        "completed_courses": ["Linear Algebra", "Statistics", "Algorithms"],
        "preferred_locations": ["Cambridge"],
        "budget_usd": 100000,
    }
    program = {
        "minimum_gpa": 3.5,
        "keywords": ["Python", "Machine Learning", "NLP"],
        "prerequisites": ["Linear Algebra", "Statistics"],
        "location": "Cambridge",
        "tuition_usd": 50000,
    }

    total, breakdown, _metadata = ProgramScorer.compute_score(user_profile, program, research_similarity=0.85)

    assert total > 70
    assert breakdown["gpa"] > 0
    assert breakdown["relevance"] > 0
    assert breakdown["prerequisites"] > 0
    assert breakdown["research_alignment"] > 0
    assert breakdown["practical_factors"] > 0


def test_low_gpa_match():
    user_profile = {"gpa_normalized": 2.5}
    program = {"minimum_gpa": 3.5}

    _total, breakdown, metadata = ProgramScorer.compute_score(user_profile, program)

    assert breakdown["gpa"] < 20.0
    assert metadata["gpa"]["status"] == "below_requirement"


def test_no_prerequisites():
    user_profile = {"completed_courses": ["Intro to CS"]}
    program = {}

    _total, breakdown, metadata = ProgramScorer.compute_score(user_profile, program)

    assert breakdown["prerequisites"] > 0
    assert metadata["prerequisites"]["reason"] == "no_prerequisites"


def test_missing_prerequisites():
    user_profile = {"completed_courses": ["Linear Algebra"]}
    program = {"prerequisites": ["Linear Algebra", "Statistics", "Calculus"]}

    _total, _breakdown, metadata = ProgramScorer.compute_score(user_profile, program)

    assert len(metadata["prerequisites"]["missing_prerequisites"]) == 2


def test_no_research_similarity():
    user_profile = {"gpa_normalized": 3.5}
    program = {"minimum_gpa": 3.0}

    _total, breakdown, _metadata = ProgramScorer.compute_score(user_profile, program, research_similarity=0.0)

    assert breakdown["research_alignment"] == 0.0
