"""Unit-test-specific fixtures and environment defaults."""

import os

import pytest

os.environ.setdefault("X_SERVICE_TOKEN", "test-service-token")


@pytest.fixture
def service_token_header():
    """Provide a stable service token header for unit tests."""
    return {"X-Service-Token": os.environ["X_SERVICE_TOKEN"]}
