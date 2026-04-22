"""Tests for program match scoring logic."""

import pytest

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


def test_explicit_llm_research_alignment_score_overrides_similarity():
    user_profile = {"gpa_normalized": 3.8}
    program = {"minimum_gpa": 3.5}

    _total, breakdown, metadata = ProgramScorer.compute_score(
        user_profile,
        program,
        research_similarity=0.2,
        research_alignment_score=87.0,
        research_alignment_metadata={
            "provider": "openai",
            "model": "gpt-test",
            "alignment_summary": "Strong methodological overlap.",
            "fallback_used": False,
        },
    )

    assert breakdown["research_alignment"] == pytest.approx(13.05)
    assert metadata["research_alignment"]["score"] == 87.0
    assert metadata["research_alignment"]["provider"] == "openai"
    assert metadata["research_alignment"]["model"] == "gpt-test"
    assert metadata["research_alignment"]["similarity"] == 0.2


def test_research_alignment_metadata_cannot_override_computed_values():
    user_profile = {"gpa_normalized": 3.8}
    program = {"minimum_gpa": 3.5}

    _total, _breakdown, metadata = ProgramScorer.compute_score(
        user_profile,
        program,
        research_similarity=0.42,
        research_alignment_score=88.0,
        research_alignment_metadata={
            "score": 12.0,
            "similarity": 0.1,
            "provider": "openai",
        },
    )

    assert metadata["research_alignment"]["score"] == 88.0
    assert metadata["research_alignment"]["similarity"] == 0.42
    assert metadata["research_alignment"]["provider"] == "openai"
