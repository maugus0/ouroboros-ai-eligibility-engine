"""FastAPI dependency for inter-service internal bearer token validation."""

import json
import time
from typing import Any, Optional, cast

import httpx
import jwt
from fastapi import HTTPException, Request
from jwt import PyJWK

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_jwks_cache_by_url: dict[str, dict] = {}


async def require_service_token(request: Request) -> None:
    """FastAPI dependency — validates internal bearer token.

    Raises:
        HTTPException: If token is missing or invalid.
    """
    if not settings.INTERNAL_TOKEN_VERIFY_ENABLED:
        return

    claims = await _decode_internal_service_token(request.headers.get("Authorization"))
    if claims is not None:
        return
    raise HTTPException(status_code=401, detail="Internal bearer token missing or invalid")


async def _decode_internal_service_token(authorization: Optional[str]) -> Optional[dict]:
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return None

    verification_key = await _resolve_internal_token_verification_key(token)
    if verification_key is None:
        logger.warning("internal_token_verification_key_unavailable")
        return None

    try:
        verification_key_typed = cast(Any, verification_key)
        return jwt.decode(
            token,
            verification_key_typed,
            algorithms=[settings.INTERNAL_TOKEN_SIGNING_ALGORITHM],
            audience=settings.INTERNAL_TOKEN_AUDIENCE,
            issuer=settings.INTERNAL_TOKEN_ISSUER,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("invalid_internal_service_token", error=str(exc))
        return None


def _normalize_key(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if "\\n" in value:
        value = value.replace("\\n", "\n")
    return value


def _decode_jwks_key(jwk: dict) -> Optional[object]:
    try:
        return PyJWK.from_json(json.dumps(jwk)).key
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("invalid_internal_jwk", error=str(exc), kid=jwk.get("kid"))
        return None


def _extract_keys_from_jwks(jwks_payload: dict) -> dict[str, object]:
    result: dict[str, object] = {}
    for entry in jwks_payload.get("keys") or []:
        kid = entry.get("kid")
        if not kid:
            continue
        parsed = _decode_jwks_key(entry)
        if parsed is not None:
            result[str(kid)] = parsed
    return result


def _resolve_configured_public_key_by_kid(kid: Optional[str]) -> Optional[str]:
    if not kid:
        return None

    configured_keys = settings.get_internal_token_public_keys()
    configured_key = configured_keys.get(str(kid))
    if not configured_key:
        return None

    return _normalize_key(str(configured_key))


async def _fetch_jwks_keys(jwks_url: str) -> dict[str, object]:
    async with httpx.AsyncClient(timeout=settings.INTERNAL_TOKEN_JWKS_TIMEOUT_SECONDS) as client:
        response = await client.get(jwks_url)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        return {}
    return _extract_keys_from_jwks(payload)


async def _resolve_jwks_key(jwks_url: str, kid: str) -> Optional[object]:
    now = time.time()
    cache = _jwks_cache_by_url.get(jwks_url)
    if cache and cache.get("expires_at", 0) > now:
        return cache.get("keys", {}).get(kid)

    try:
        fetched_keys = await _fetch_jwks_keys(jwks_url)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("internal_jwks_fetch_failed", error=str(exc), jwks_url=jwks_url)
        fetched_keys = cache.get("keys", {}) if cache else {}

    _jwks_cache_by_url[jwks_url] = {
        "keys": fetched_keys,
        "expires_at": now + max(settings.INTERNAL_TOKEN_JWKS_REFRESH_SECONDS, 1),
    }
    return fetched_keys.get(kid)


async def _resolve_internal_token_verification_key(token: str) -> Optional[Any]:
    try:
        token_header = jwt.get_unverified_header(token)
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    kid = token_header.get("kid")
    algorithm = str(settings.INTERNAL_TOKEN_SIGNING_ALGORITHM or "").upper()
    resolved_key: Optional[Any] = None

    if algorithm.startswith("HS"):
        if settings.INTERNAL_TOKEN_PUBLIC_KEY:
            resolved_key = _normalize_key(settings.INTERNAL_TOKEN_PUBLIC_KEY)
    else:
        configured_kid_key = _resolve_configured_public_key_by_kid(str(kid) if kid else None)
        if configured_kid_key:
            resolved_key = configured_kid_key
        elif kid and settings.INTERNAL_TOKEN_JWKS_URL:
            resolved_key = await _resolve_jwks_key(settings.INTERNAL_TOKEN_JWKS_URL, str(kid))

        if resolved_key is None and settings.INTERNAL_TOKEN_PUBLIC_KEY:
            resolved_key = _normalize_key(settings.INTERNAL_TOKEN_PUBLIC_KEY)

    return resolved_key
