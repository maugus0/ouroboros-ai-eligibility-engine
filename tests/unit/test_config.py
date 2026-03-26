"""Tests for application configuration."""

# pylint: disable=import-outside-toplevel

import os


def test_settings_load():
    os.environ.setdefault("ALLOW_DB_FAILURE", "true")
    os.environ.setdefault("X_SERVICE_TOKEN", "test-service-token")

    from app.config import settings

    assert settings.DB_NAME == "ouroboros_eligibility_db"
    assert settings.DB_PORT == 5432
    assert settings.DB_POOL_MIN_SIZE == 5
    assert settings.DB_POOL_MAX_SIZE == 20


def test_settings_db_helpers():
    os.environ.setdefault("ALLOW_DB_FAILURE", "true")
    os.environ.setdefault("X_SERVICE_TOKEN", "test-service-token")

    from app.config import settings

    assert isinstance(settings.get_db_host(), str)
    assert isinstance(settings.get_db_port(), int)
    assert isinstance(settings.get_db_name(), str)
    assert isinstance(settings.get_db_user(), str)


def test_scoring_weight_defaults():
    os.environ.setdefault("X_SERVICE_TOKEN", "test-service-token")
    from app.config import settings

    program_sum = (
        settings.PROGRAM_WEIGHT_GPA
        + settings.PROGRAM_WEIGHT_RELEVANCE
        + settings.PROGRAM_WEIGHT_PREREQUISITES
        + settings.PROGRAM_WEIGHT_RESEARCH
        + settings.PROGRAM_WEIGHT_PRACTICAL
    )
    assert program_sum == 100

    scholarship_sum = (
        settings.SCHOLARSHIP_WEIGHT_ELIGIBILITY
        + settings.SCHOLARSHIP_WEIGHT_PREFERRED
        + settings.SCHOLARSHIP_WEIGHT_FUNDING
        + settings.SCHOLARSHIP_WEIGHT_COMPETITION
    )
    assert scholarship_sum == 100


def test_vector_search_settings():
    os.environ.setdefault("X_SERVICE_TOKEN", "test-service-token")
    from app.config import settings

    assert settings.EMBEDDING_DIMENSION == 1536
    assert settings.VECTOR_SIMILARITY_TOP_K > 0
    assert 0.0 <= settings.VECTOR_SIMILARITY_THRESHOLD <= 1.0
