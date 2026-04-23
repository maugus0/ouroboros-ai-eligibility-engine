"""Unit-test-specific fixtures and environment defaults."""

import pytest


@pytest.fixture
def service_token_header(internal_token_header):
    """Backward-compatible alias for unit tests still using the legacy fixture name."""
    return internal_token_header
