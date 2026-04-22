"""Unit tests for batch matching orchestration."""

# pylint: disable=protected-access

import asyncio

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


@pytest.mark.asyncio
async def test_evaluate_batch_preserves_input_order_under_concurrency(monkeypatch):
    async def fake_evaluate(
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        user_profile: dict,
        entity_data: dict,
        include_attribution: bool = True,
    ):
        _ = (self, user_id, entity_type, user_profile, entity_data, include_attribution)
        await asyncio.sleep(0.02 if entity_id == "slow" else 0.0)
        return {
            "match_result": {
                "id": f"match-{entity_id}",
                "entity_id": entity_id,
                "match_score": 70.0,
            }
        }

    monkeypatch.setattr(MatchingService, "evaluate", fake_evaluate)

    service = MatchingService()
    result = await service.evaluate_batch(
        user_id="user-1",
        user_profile={"gpa_normalized": 3.9},
        evaluations=[
            {"entity_type": "scholarship", "entity_id": "slow", "entity_data": {}},
            {"entity_type": "scholarship", "entity_id": "fast", "entity_data": {}},
        ],
    )

    assert [item["entity_id"] for item in result["results"]] == ["slow", "fast"]


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


@pytest.mark.asyncio
async def test_evaluate_does_not_persist_embedding_model_as_llm_model(monkeypatch):
    captured_match_data = {}

    async def fake_get_research_alignment(
        self,
        user_id: str,
        program_id: str,
        user_profile: dict,
        program: dict,
    ):
        _ = (self, user_id, program_id, user_profile, program)
        return {
            "score": 84.0,
            "similarity": 0.73,
            "provider": "pgvector_fallback",
            "model": None,
            "embedding_model": "text-embedding-3-small",
            "fallback_used": True,
        }

    def fake_compute_score(
        user_profile: dict,
        program: dict,
        research_similarity: float,
        research_alignment_score: float,
        research_alignment_metadata: dict,
    ):
        _ = (user_profile, program, research_similarity, research_alignment_metadata)
        return 88.0, {"research_alignment": 12.6}, {"research_alignment": {"score": research_alignment_score}}

    async def fake_match_create(self, data: dict):
        _ = self
        captured_match_data.update(data)
        return {
            "id": "match-1",
            **data,
        }

    async def fake_history_create(self, data: dict):
        _ = (self, data)
        return {"id": "history-1"}

    monkeypatch.setattr(MatchingService, "_get_research_alignment", fake_get_research_alignment)
    monkeypatch.setattr("app.services.matching_service.ProgramScorer.compute_score", fake_compute_score)
    monkeypatch.setattr("app.services.matching_service.MatchResultRepository.create", fake_match_create)
    monkeypatch.setattr("app.services.matching_service.ScoringHistoryRepository.create", fake_history_create)

    service = MatchingService()
    result = await service.evaluate(
        user_id="user-1",
        entity_type="program",
        entity_id="program-1",
        user_profile={"research_interests": "NLP"},
        entity_data={"research_focus": "NLP"},
        include_attribution=False,
    )

    assert result["match_result"]["id"] == "match-1"
    assert captured_match_data["llm_model_used"] is None
    assert captured_match_data["llm_fallback_used"] is True
