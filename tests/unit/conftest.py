"""Unit-test-specific fixtures and environment defaults."""

import pytest


@pytest.fixture
def service_token_header(request):
    """Backward-compatible alias for unit tests still using the legacy fixture name."""
    return request.getfixturevalue("internal_token_header")
