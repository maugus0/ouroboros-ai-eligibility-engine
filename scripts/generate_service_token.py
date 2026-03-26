"""Generate a secure X-Service-Token for inter-service authentication."""

import secrets
import sys


def generate_token(length: int = 64) -> str:
    """Generate a cryptographically secure token."""
    return secrets.token_urlsafe(length)


if __name__ == "__main__":
    token_length = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    token = generate_token(token_length)
    print(f"Generated X-Service-Token:\n{token}")
    print(f"\nAdd to your .env file:")
    print(f"X_SERVICE_TOKEN={token}")
