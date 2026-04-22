"""Tests for health endpoints."""

from fastapi.testclient import TestClient

from app.api import health as health_api
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["message"] == "Eligibility Engine"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_endpoint_reports_not_connected_when_pool_missing(monkeypatch):
    def raise_runtime_error():
        raise RuntimeError("pool not initialised")

    monkeypatch.setattr(health_api, "get_pool", raise_runtime_error)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["database"] == "not_connected"


def test_health_endpoint_reports_connected_when_query_succeeds(monkeypatch):
    class FakeConnection:
        async def execute(self, query: str):
            assert query == "SELECT 1"
            return "SELECT 1"

    class FakeAcquireContext:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        def acquire(self):
            return FakeAcquireContext()

    def get_fake_pool():
        return FakePool()

    monkeypatch.setattr(health_api, "get_pool", get_fake_pool)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["database"] == "connected"
