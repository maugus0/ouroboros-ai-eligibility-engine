"""Tests for LLM pipeline structured research alignment handling."""

import pytest

from app.services.llm_service import LLMPipelineService


def test_parse_research_alignment_response_accepts_valid_json():
    content = """
    {
      "score": 87,
      "alignment_summary": "Strong overlap in NLP and multilingual systems.",
      "overlapping_themes": ["NLP", "multilingual AI"],
      "unique_student_interests": ["speech"],
      "recommended_faculty": ["Prof. Ada"],
      "confidence": "high"
    }
    """

    parsed = LLMPipelineService._parse_research_alignment_response(content)  # pylint: disable=protected-access

    assert parsed["score"] == 87.0
    assert parsed["alignment_summary"].startswith("Strong overlap")
    assert parsed["overlapping_themes"] == ["NLP", "multilingual AI"]
    assert parsed["confidence"] == "high"


def test_parse_research_alignment_response_clamps_out_of_range_scores():
    parsed = LLMPipelineService._parse_research_alignment_response(  # pylint: disable=protected-access
        '{"score": 120, "alignment_summary": "Excellent", "confidence": "high"}'
    )

    assert parsed["score"] == 100.0


def test_parse_research_alignment_response_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        LLMPipelineService._parse_research_alignment_response("not-json")  # pylint: disable=protected-access


def test_parse_research_alignment_response_wraps_scalar_strings_for_list_fields():
    parsed = LLMPipelineService._parse_research_alignment_response(  # pylint: disable=protected-access
        (
            '{"score": 72, "alignment_summary": "Good", '
            '"overlapping_themes": "nlp", "unique_student_interests": "robotics", '
            '"recommended_faculty": "Prof. X"}'
        )
    )

    assert parsed["overlapping_themes"] == ["nlp"]
    assert parsed["unique_student_interests"] == ["robotics"]
    assert parsed["recommended_faculty"] == ["Prof. X"]


def test_parse_attribution_response_wraps_scalar_strings_for_strengths_and_gaps():
    parsed = LLMPipelineService._parse_attribution_response(  # pylint: disable=protected-access
        '{"strengths": "Strong GPA", "gaps": "Needs more research", "reasoning": "Test"}'
    )

    assert parsed["strengths"] == ["Strong GPA"]
    assert parsed["gaps"] == ["Needs more research"]
