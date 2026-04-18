"""Tests for matching and internal eligibility evaluation endpoints."""

from fastapi.testclient import TestClient

from app.main import app
from app.services.matching_service import MatchingService

client = TestClient(app)


def test_internal_evaluate_alias_supports_single_request(service_token_header, monkeypatch):
    async def fake_evaluate(
        _self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        user_profile: dict,
        entity_data: dict,
        include_attribution: bool = True,
    ):
        _ = (user_profile, entity_data, include_attribution)
        return {
            "match_result": {
                "id": "match-1",
                "user_id": user_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "match_score": 88.0,
            }
        }

    monkeypatch.setattr(MatchingService, "evaluate", fake_evaluate)

    response = client.post(
        "/api/v1/eligibility/evaluate",
        headers=service_token_header,
        json={
            "user_id": "user-1",
            "entity_type": "scholarship",
            "entity_id": "scholarship-1",
            "user_profile": {"gpa_normalized": 3.8},
            "entity_data": {"minimum_gpa": 3.5},
            "include_attribution": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "Evaluation complete"
    assert body["data"]["match_result"]["entity_type"] == "scholarship"
    assert body["data"]["match_result"]["entity_id"] == "scholarship-1"


def test_internal_evaluate_alias_supports_batch_request(service_token_header, monkeypatch):
    async def fake_evaluate_batch(
        _self,
        user_id: str,
        user_profile: dict,
        evaluations: list[dict],
        include_attribution: bool = True,
    ):
        _ = (user_profile, include_attribution)
        return {
            "count": len(evaluations),
            "results": [
                {
                    "entity_type": item["entity_type"],
                    "entity_id": item["entity_id"],
                    "match_result": {
                        "id": f"match-{index}",
                        "user_id": user_id,
                        "entity_type": item["entity_type"],
                        "entity_id": item["entity_id"],
                        "match_score": 70.0 + index,
                    },
                }
                for index, item in enumerate(evaluations, start=1)
            ],
        }

    monkeypatch.setattr(MatchingService, "evaluate_batch", fake_evaluate_batch)

    response = client.post(
        "/api/v1/eligibility/evaluate/batch",
        headers=service_token_header,
        json={
            "user_id": "user-1",
            "user_profile": {"gpa_normalized": 3.8},
            "evaluations": [
                {
                    "entity_type": "scholarship",
                    "entity_id": "scholarship-1",
                    "entity_data": {"minimum_gpa": 3.5},
                },
                {
                    "entity_type": "scholarship",
                    "entity_id": "scholarship-2",
                    "entity_data": {"minimum_gpa": 3.2},
                    "include_attribution": False,
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "Batch evaluation complete"
    assert body["data"]["count"] == 2
    assert [item["match_result"]["entity_id"] for item in body["data"]["results"]] == [
        "scholarship-1",
        "scholarship-2",
    ]
