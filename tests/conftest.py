"""Pytest configuration and shared fixtures."""

import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest

INTERNAL_TEST_KEY = "internal-test-signing-key-with-32-bytes"

os.environ.setdefault("ALLOW_DB_FAILURE", "true")
os.environ.setdefault("USE_MOCK_DATA", "true")
os.environ.setdefault("INTERNAL_TOKEN_VERIFY_ENABLED", "true")
os.environ.setdefault("INTERNAL_TOKEN_SIGNING_ALGORITHM", "HS256")
os.environ.setdefault("INTERNAL_TOKEN_PUBLIC_KEY", INTERNAL_TEST_KEY)
os.environ.setdefault("INTERNAL_TOKEN_ISSUER", "ouroboros-orchestrator-internal")
os.environ.setdefault("INTERNAL_TOKEN_AUDIENCE", "ouroboros.eligibility-engine")

from app.config import settings  # noqa: E402  # pylint: disable=wrong-import-position


@pytest.fixture
def mock_settings():
    return {
        "DB_HOST": "localhost",
        "DB_NAME": "test_db",
        "USE_MOCK_DATA": True,
        "ALLOW_DB_FAILURE": True,
        "INTERNAL_TOKEN_VERIFY_ENABLED": settings.INTERNAL_TOKEN_VERIFY_ENABLED,
        "INTERNAL_TOKEN_SIGNING_ALGORITHM": settings.INTERNAL_TOKEN_SIGNING_ALGORITHM,
        "INTERNAL_TOKEN_PUBLIC_KEY": settings.INTERNAL_TOKEN_PUBLIC_KEY,
        "INTERNAL_TOKEN_AUDIENCE": settings.INTERNAL_TOKEN_AUDIENCE,
        "INTERNAL_TOKEN_ISSUER": settings.INTERNAL_TOKEN_ISSUER,
    }


@pytest.fixture
def service_token_header():
    """Generate a valid internal bearer token for tests."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "user-123",
        "aud": settings.INTERNAL_TOKEN_AUDIENCE,
        "iss": settings.INTERNAL_TOKEN_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "sid": "session-123",
        "trace_id": "trace-123",
        "jti": "jti-123",
    }
    token = jwt.encode(payload, settings.INTERNAL_TOKEN_PUBLIC_KEY, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}
