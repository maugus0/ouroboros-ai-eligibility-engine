"""Deprecated helper.

X-Service-Token auth has been removed. Use the internal bearer-token settings
in `.env` instead: INTERNAL_TOKEN_VERIFY_ENABLED, INTERNAL_TOKEN_SIGNING_ALGORITHM,
INTERNAL_TOKEN_PUBLIC_KEY, INTERNAL_TOKEN_ISSUER, and INTERNAL_TOKEN_AUDIENCE.
"""

if __name__ == "__main__":
    print("X-Service-Token auth has been removed. Configure internal bearer-token settings in .env instead.")
