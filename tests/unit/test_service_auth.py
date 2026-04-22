"""Unit tests for internal token key resolution in service_auth middleware."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.middleware import service_auth


@pytest.fixture(autouse=True)
def reset_jwks_cache() -> None:
    service_auth._jwks_cache_by_url.clear()  # pylint: disable=protected-access


def _build_token_with_kid(kid: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "internal-user",
        "aud": "ouroboros.eligibility-engine",
        "iss": "ouroboros-orchestrator-internal",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    return jwt.encode(payload, "internal-test-signing-key-with-32-bytes", algorithm="HS256", headers={"kid": kid})


@pytest.mark.asyncio
async def test_fetch_jwks_keys_uses_httpx_client_and_extracts_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure JWKS is fetched via AsyncClient.get and extraction receives parsed payload."""
    observed: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"keys": [{"kid": "kid-1", "kty": "RSA"}]}

    class DummyAsyncClient:
        def __init__(self, *, timeout: int) -> None:
            observed["timeout"] = timeout

        async def __aenter__(self) -> "DummyAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> DummyResponse:
            observed["url"] = url
            return DummyResponse()

    def fake_extract(payload: dict) -> dict[str, object]:
        observed["payload"] = payload
        return {"kid-1": "parsed-key"}

    monkeypatch.setattr(service_auth.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(service_auth, "_extract_keys_from_jwks", fake_extract)
    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_JWKS_TIMEOUT_SECONDS", 7)

    result = await service_auth._fetch_jwks_keys(
        "https://issuer.example/.well-known/jwks.json"
    )  # pylint: disable=protected-access

    assert result == {"kid-1": "parsed-key"}
    assert observed["url"] == "https://issuer.example/.well-known/jwks.json"
    assert observed["timeout"] == 7
    assert observed["payload"] == {"keys": [{"kid": "kid-1", "kty": "RSA"}]}


@pytest.mark.asyncio
async def test_resolve_jwks_key_uses_cache_before_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def fake_fetch(_jwks_url: str) -> dict[str, object]:
        calls["count"] += 1
        return {"kid-1": "fetched-key"}

    monkeypatch.setattr(service_auth, "_fetch_jwks_keys", fake_fetch)
    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_JWKS_REFRESH_SECONDS", 60)

    key_first = await service_auth._resolve_jwks_key(
        "https://issuer.example/jwks", "kid-1"
    )  # pylint: disable=protected-access
    key_second = await service_auth._resolve_jwks_key(
        "https://issuer.example/jwks", "kid-1"
    )  # pylint: disable=protected-access

    assert key_first == "fetched-key"
    assert key_second == "fetched-key"
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_resolve_jwks_key_falls_back_to_cached_keys_when_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    jwks_url = "https://issuer.example/jwks"
    service_auth._jwks_cache_by_url[jwks_url] = {  # pylint: disable=protected-access
        "keys": {"kid-1": "cached-key"},
        "expires_at": 0,
    }

    async def failing_fetch(_jwks_url: str) -> dict[str, object]:
        raise RuntimeError("network down")

    monkeypatch.setattr(service_auth, "_fetch_jwks_keys", failing_fetch)
    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_JWKS_REFRESH_SECONDS", 30)

    key = await service_auth._resolve_jwks_key(jwks_url, "kid-1")  # pylint: disable=protected-access

    assert key == "cached-key"
    assert (
        service_auth._jwks_cache_by_url[jwks_url]["keys"]["kid-1"] == "cached-key"
    )  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_resolve_internal_verification_key_prefers_configured_kid_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _build_token_with_kid("kid-1")

    async def jwks_should_not_be_used(_jwks_url: str, _kid: str) -> object:
        raise AssertionError("JWKS should not be queried when kid mapping exists")

    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_SIGNING_ALGORITHM", "RS256")
    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_JWKS_URL", "https://issuer.example/jwks")
    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_PUBLIC_KEY", "")
    monkeypatch.setattr(
        service_auth,
        "_resolve_configured_public_key_by_kid",
        lambda kid: "-----BEGIN PUBLIC KEY-----\nabc\n-----END PUBLIC KEY-----" if kid == "kid-1" else None,
    )
    monkeypatch.setattr(service_auth, "_resolve_jwks_key", jwks_should_not_be_used)

    key = await service_auth._resolve_internal_token_verification_key(token)  # pylint: disable=protected-access

    assert key == "-----BEGIN PUBLIC KEY-----\nabc\n-----END PUBLIC KEY-----"


@pytest.mark.asyncio
async def test_resolve_internal_verification_key_uses_jwks_for_kid_when_mapping_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _build_token_with_kid("kid-2")

    async def fake_resolve_jwks_key(_jwks_url: str, _kid: str) -> object:
        return "jwks-key"

    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_SIGNING_ALGORITHM", "RS256")
    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_JWKS_URL", "https://issuer.example/jwks")
    monkeypatch.setattr(service_auth.settings, "INTERNAL_TOKEN_PUBLIC_KEY", "")
    monkeypatch.setattr(service_auth, "_resolve_configured_public_key_by_kid", lambda _kid: None)
    monkeypatch.setattr(service_auth, "_resolve_jwks_key", fake_resolve_jwks_key)

    key = await service_auth._resolve_internal_token_verification_key(token)  # pylint: disable=protected-access

    assert key == "jwks-key"
