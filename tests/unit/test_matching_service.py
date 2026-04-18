"""Unit tests for batch matching orchestration."""

# pylint: disable=protected-access

import pytest

from app.services.matching_service import MatchingService


@pytest.mark.asyncio
async def test_evaluate_batch_returns_count_and_per_entity_results(monkeypatch):
    captured_include_attribution: list[bool] = []

    async def fake_evaluate(
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        user_profile: dict,
        entity_data: dict,
        include_attribution: bool = True,
    ):
        _ = (self, user_profile, entity_data)
        captured_include_attribution.append(include_attribution)
        return {
            "match_result": {
                "id": f"match-{entity_id}",
                "user_id": user_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "match_score": 80.0,
            }
        }

    monkeypatch.setattr(MatchingService, "evaluate", fake_evaluate)

    service = MatchingService()
    result = await service.evaluate_batch(
        user_id="user-1",
        user_profile={"gpa_normalized": 3.9},
        evaluations=[
            {
                "entity_type": "scholarship",
                "entity_id": "scholarship-1",
                "entity_data": {"minimum_gpa": 3.5},
                "include_attribution": None,
            },
            {
                "entity_type": "scholarship",
                "entity_id": "scholarship-2",
                "entity_data": {"minimum_gpa": 3.6},
                "include_attribution": False,
            },
        ],
        include_attribution=True,
    )

    assert result["count"] == 2
    assert [item["entity_id"] for item in result["results"]] == ["scholarship-1", "scholarship-2"]
    assert result["results"][0]["match_result"]["match_score"] == 80.0
    assert captured_include_attribution == [True, False]


def test_extract_program_research_focus_prefers_research_specific_fields():
    program = {
        "research_focus": "Multilingual NLP and responsible AI",
        "faculty_research": ["Large language models"],
        "keywords": ["ignore-me-only-if-no-research-focus"],
    }

    result = MatchingService._extract_program_research_focus(program)

    assert "Multilingual NLP and responsible AI" in result
    assert "Large language models" in result


def test_combine_research_alignment_scores_blends_vector_and_llm():
    combined = MatchingService._combine_research_alignment_scores(vector_similarity=0.8, llm_score=90.0)

    assert combined == pytest.approx(86.0)
