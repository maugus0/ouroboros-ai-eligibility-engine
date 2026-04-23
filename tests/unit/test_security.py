"""Tests for internal bearer token validation."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_missing_service_token():
    response = client.post("/matching/evaluate", json={"user_id": "test"})
    assert response.status_code == 401


def test_invalid_service_token():
    response = client.post(
        "/matching/evaluate",
        json={"user_id": "test"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_valid_service_token_returns_non_401(internal_token_header):
    response = client.post(
        "/matching/evaluate",
        json={"user_id": "test"},
        headers=internal_token_header,
    )
    assert response.status_code not in (401, 403)


def test_missing_token_on_results():
    response = client.get("/matching/results/user-123")
    assert response.status_code == 401


def test_missing_token_on_attribution():
    response = client.get("/attribution/report/match-123")
    assert response.status_code == 401
