"""Unit tests for the explainability / attribution pipeline."""

from unittest.mock import AsyncMock

import pytest

from app.services.explainability_service import ExplainabilityService


@pytest.mark.asyncio
async def test_generate_report_combines_rule_and_llm_signals(monkeypatch):
    service = ExplainabilityService()
    captured_report = {}

    async def fake_generate_attribution(**_kwargs):
        return {
            "strengths": ["Strong research alignment with faculty methods"],
            "gaps": ["Need stronger publication record"],
            "reasoning": "The student matches strongly on academics and research direction.",
            "recommendations": [{"action": "Build a publication pipeline", "priority": "high"}],
            "confidence": "medium",
            "provider": "openai",
            "model": "gpt-test",
            "fallback_used": False,
        }

    async def fake_create(report_data):
        captured_report.update(report_data)
        return {"id": "report-1", **report_data}

    monkeypatch.setattr(service.llm_service, "generate_attribution", fake_generate_attribution)
    monkeypatch.setattr(service.attribution_repo, "create", fake_create)

    report = await service.generate_report(
        match_id="match-1",
        user_profile={"completed_courses": ["Algorithms"]},
        entity_data={"prerequisites": ["Algorithms"], "entity_type": "program"},
        score_breakdown={
            "gpa_threshold": 18.0,
            "field_relevance": 28.0,
            "prerequisite_match": 22.0,
            "research_alignment": 14.0,
            "practical_factors": 8.0,
        },
        match_score=90.0,
    )

    assert report is not None
    assert captured_report["llm_provider"] == "openai"
    assert captured_report["llm_model"] == "gpt-test"
    assert captured_report["confidence"] == "medium"
    assert "Overall strong match (90.0/100)" in captured_report["strengths"]
    assert "Strong research alignment with faculty methods" in captured_report["strengths"]
    assert "Need stronger publication record" in captured_report["gaps"]
    assert captured_report["reasoning"].startswith("The student matches strongly")
    assert captured_report["recommendations"][0]["priority"] == "high"


@pytest.mark.asyncio
async def test_generate_report_falls_back_to_rule_based_reasoning(monkeypatch):
    service = ExplainabilityService()

    monkeypatch.setattr(
        service.llm_service,
        "generate_attribution",
        AsyncMock(side_effect=RuntimeError("llm unavailable")),
    )
    monkeypatch.setattr(
        service.attribution_repo,
        "create",
        AsyncMock(side_effect=lambda report_data: {"id": "report-2", **report_data}),
    )

    report = await service.generate_report(
        match_id="match-2",
        user_profile={"completed_courses": []},
        entity_data={"prerequisites": ["Statistics"], "entity_type": "program"},
        score_breakdown={
            "gpa_threshold": 8.0,
            "field_relevance": 6.0,
            "prerequisite_match": 0.0,
            "research_alignment": 4.0,
            "practical_factors": 2.0,
        },
        match_score=20.0,
    )

    assert report is not None
    assert report["llm_provider"] == "rule_based"
    assert report["confidence"] == "low"
    assert "Match score: 20.0/100." in report["reasoning"]
    assert any("Missing prerequisite: Statistics" == gap for gap in report["gaps"])


@pytest.mark.asyncio
async def test_generate_report_uses_mandatory_failure_metadata_for_scholarship_gaps(monkeypatch):
    service = ExplainabilityService()

    monkeypatch.setattr(
        service.llm_service,
        "generate_attribution",
        AsyncMock(side_effect=RuntimeError("llm unavailable")),
    )
    monkeypatch.setattr(
        service.attribution_repo,
        "create",
        AsyncMock(side_effect=lambda report_data: {"id": "report-3", **report_data}),
    )

    report = await service.generate_report(
        match_id="match-3",
        user_profile={},
        entity_data={"entity_type": "scholarship"},
        score_breakdown={
            "eligibility": 0.0,
            "preferred_criteria": 0.0,
            "funding_coverage": 0.0,
            "competition_estimate": 0.0,
        },
        match_score=0.0,
        score_metadata={
            "eligibility": {
                "mandatory_passed": False,
                "failed_criteria": [
                    {
                        "criterion": "gpa",
                        "message": "GPA 2.8 is below the minimum requirement of 3.5.",
                    }
                ],
            }
        },
    )

    assert report is not None
    assert report["llm_provider"] == "rule_based"
    assert report["gaps"] == ["GPA 2.8 is below the minimum requirement of 3.5."]
    assert "GPA 2.8 is below the minimum requirement of 3.5." in report["reasoning"]


@pytest.mark.asyncio
async def test_generate_report_preserves_empty_llm_recommendations(monkeypatch):
    service = ExplainabilityService()

    async def fake_generate_attribution(**_kwargs):
        return {
            "strengths": [],
            "gaps": [],
            "reasoning": "No additional actions recommended.",
            "recommendations": [],
            "confidence": "high",
            "provider": "openai",
            "model": "gpt-test",
            "fallback_used": False,
        }

    monkeypatch.setattr(service.llm_service, "generate_attribution", fake_generate_attribution)
    monkeypatch.setattr(
        service.attribution_repo,
        "create",
        AsyncMock(side_effect=lambda report_data: {"id": "report-4", **report_data}),
    )

    report = await service.generate_report(
        match_id="match-4",
        user_profile={"completed_courses": []},
        entity_data={"prerequisites": ["Statistics"], "entity_type": "program"},
        score_breakdown={
            "gpa_threshold": 8.0,
            "field_relevance": 6.0,
            "prerequisite_match": 0.0,
            "research_alignment": 4.0,
            "practical_factors": 2.0,
        },
        match_score=20.0,
    )

    assert report is not None
    assert report["recommendations"] == []
