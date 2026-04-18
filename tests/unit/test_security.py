"""Tests for X-Service-Token validation."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_missing_service_token():
    response = client.post("/matching/evaluate", json={"user_id": "test"})
    assert response.status_code == 401


def test_missing_service_token_on_internal_evaluate():
    response = client.post("/api/v1/eligibility/evaluate", json={"user_id": "test"})
    assert response.status_code == 401


def test_missing_service_token_on_batch_evaluate():
    response = client.post("/api/v1/eligibility/evaluate/batch", json={"user_id": "test"})
    assert response.status_code == 401


def test_invalid_service_token():
    response = client.post(
        "/matching/evaluate",
        json={"user_id": "test"},
        headers={"X-Service-Token": "wrong-token"},
    )
    assert response.status_code == 403


def test_valid_service_token_returns_non_401(service_token_header):
    response = client.post(
        "/matching/evaluate",
        json={"user_id": "test"},
        headers=service_token_header,
    )
    assert response.status_code not in (401, 403)


def test_missing_token_on_results():
    response = client.get("/matching/results/user-123")
    assert response.status_code == 401


def test_missing_token_on_attribution():
    response = client.get("/attribution/report/match-123")
    assert response.status_code == 401


def test_missing_token_on_internal_eligibility_report():
    response = client.get("/api/v1/eligibility/report/match-123")
    assert response.status_code == 401


def test_missing_token_on_internal_eligibility_evaluate():
    response = client.post("/api/v1/eligibility/evaluate", json={"user_id": "test"})
    assert response.status_code == 401
