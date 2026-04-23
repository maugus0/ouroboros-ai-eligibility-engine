"""Tests for attribution report endpoints."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.main import app
from app.services.explainability_service import ExplainabilityService

client = TestClient(app)


def test_internal_report_alias_returns_report(service_token_header, monkeypatch):
    async def fake_get_report(_self, match_id: str):
        return {
            "id": "report-1",
            "match_id": match_id,
            "strengths": ["Strong GPA score"],
            "gaps": [],
            "reasoning": "Strong fit.",
            "confidence": "high",
            "recommendations": [],
            "llm_provider": "openai",
            "llm_model": "gpt-test",
            "prompt_version": "attribution_report_v1",
        }

    monkeypatch.setattr(ExplainabilityService, "get_report", fake_get_report)

    response = client.get("/api/v1/eligibility/report/match-123", headers=service_token_header)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["match_id"] == "match-123"


def test_internal_report_alias_handles_missing_report(service_token_header, monkeypatch):
    monkeypatch.setattr(ExplainabilityService, "get_report", AsyncMock(return_value=None))

    response = client.get("/api/v1/eligibility/report/missing", headers=service_token_header)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["message"] == "Attribution report not found"
