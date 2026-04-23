"""Application configuration loaded from environment variables."""

import json
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "0.1.0"


class Settings(BaseSettings):
    """All application settings. Loaded from .env file."""

    # ========== Database (PostgreSQL) ==========
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "ouroboros_eligibility_db"
    DB_USERNAME: str = "postgres"
    DB_PASSWORD: str = ""

    DB_POOL_MIN_SIZE: int = 5
    DB_POOL_MAX_SIZE: int = 20
    DB_CONNECTION_TIMEOUT: int = 30

    # ========== Inter-Service Auth ==========
    INTERNAL_TOKEN_VERIFY_ENABLED: bool = False
    INTERNAL_TOKEN_SIGNING_ALGORITHM: str = "RS256"
    INTERNAL_TOKEN_PUBLIC_KEY: str = ""
    INTERNAL_TOKEN_PUBLIC_KEYS: str = "{}"
    INTERNAL_TOKEN_JWKS_URL: str = ""
    INTERNAL_TOKEN_JWKS_REFRESH_SECONDS: int = 60
    INTERNAL_TOKEN_JWKS_TIMEOUT_SECONDS: int = 2
    INTERNAL_TOKEN_AUDIENCE: str = "ouroboros.eligibility-engine"
    INTERNAL_TOKEN_ISSUER: str = "ouroboros-orchestrator-internal"

    # ========== LLM Configuration ==========
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 1500
    OPENAI_TEMPERATURE: float = 0.1

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    ANTHROPIC_MAX_TOKENS: int = 1500

    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_DELAY: int = 2

    # ========== Scoring Configuration ==========
    # Program Match Weights (must sum to 100)
    PROGRAM_WEIGHT_GPA: int = 20
    PROGRAM_WEIGHT_RELEVANCE: int = 30
    PROGRAM_WEIGHT_PREREQUISITES: int = 25
    PROGRAM_WEIGHT_RESEARCH: int = 15
    PROGRAM_WEIGHT_PRACTICAL: int = 10

    # Scholarship Match Weights (must sum to 100)
    SCHOLARSHIP_WEIGHT_ELIGIBILITY: int = 40
    SCHOLARSHIP_WEIGHT_PREFERRED: int = 30
    SCHOLARSHIP_WEIGHT_FUNDING: int = 20
    SCHOLARSHIP_WEIGHT_COMPETITION: int = 10

    # Thresholds
    MATCH_SCORE_THRESHOLD: float = 50.0
    HIGH_CONFIDENCE_THRESHOLD: float = 75.0

    # ========== Vector Search Configuration ==========
    EMBEDDING_DIMENSION: int = 1536
    VECTOR_SIMILARITY_TOP_K: int = 5
    VECTOR_SIMILARITY_THRESHOLD: float = 0.7

    # ========== Application ==========
    LOG_LEVEL: str = "INFO"
    USE_MOCK_DATA: bool = True
    ALLOW_DB_FAILURE: bool = False

    # ========== Docker ==========
    DOCKER_POSTGRES_PORT: int = 5433

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    @staticmethod
    def _get_env(primary_key: str, secondary_key: str, default_value: str) -> str:
        """Prefer app runtime `DB_*` vars, then fallback `POSTGRES_*`, then defaults."""
        return os.getenv(primary_key, os.getenv(secondary_key, default_value))

    def get_db_host(self) -> str:
        return self._get_env("DB_HOST", "POSTGRES_HOST", self.DB_HOST)

    def get_db_name(self) -> str:
        return self._get_env("DB_NAME", "POSTGRES_DATABASE", self.DB_NAME)

    def get_db_user(self) -> str:
        return self._get_env("DB_USERNAME", "POSTGRES_USER", self.DB_USERNAME)

    def get_db_password(self) -> str:
        return self._get_env("DB_PASSWORD", "POSTGRES_PASSWORD", self.DB_PASSWORD)

    def get_db_port(self) -> int:
        val = os.getenv("DB_PORT", os.getenv("POSTGRES_PORT"))
        return int(val) if val is not None else self.DB_PORT

    def validate_weights(self):
        """Validate that scoring weights sum to 100."""
        program_sum = (
            self.PROGRAM_WEIGHT_GPA
            + self.PROGRAM_WEIGHT_RELEVANCE
            + self.PROGRAM_WEIGHT_PREREQUISITES
            + self.PROGRAM_WEIGHT_RESEARCH
            + self.PROGRAM_WEIGHT_PRACTICAL
        )
        scholarship_sum = (
            self.SCHOLARSHIP_WEIGHT_ELIGIBILITY
            + self.SCHOLARSHIP_WEIGHT_PREFERRED
            + self.SCHOLARSHIP_WEIGHT_FUNDING
            + self.SCHOLARSHIP_WEIGHT_COMPETITION
        )
        if program_sum != 100:
            raise ValueError(f"Program weights must sum to 100, got {program_sum}")
        if scholarship_sum != 100:
            raise ValueError(f"Scholarship weights must sum to 100, got {scholarship_sum}")

    def get_internal_token_public_keys(self) -> dict[str, str]:
        """Parse INTERNAL_TOKEN_PUBLIC_KEYS JSON string into a kid->PEM dict."""
        try:
            parsed = json.loads(self.INTERNAL_TOKEN_PUBLIC_KEYS)
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            str(kid).strip(): str(pem).strip()
            for kid, pem in parsed.items()
            if kid is not None and str(kid).strip() and pem is not None and str(pem).strip()
        }


settings = Settings()
settings.validate_weights()
