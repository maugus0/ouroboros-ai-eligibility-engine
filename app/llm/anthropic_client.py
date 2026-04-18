"""Anthropic client with retry logic."""

from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None


def get_anthropic_client():
    """Lazy-initialise and return the Anthropic client."""
    global _client  # pylint: disable=global-statement
    if _client is None:
        from anthropic import AsyncAnthropic  # pylint: disable=import-outside-toplevel

        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def call_anthropic(
    prompt: str,
    system_message: str = "",
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Call Anthropic API with retry logic.

    Returns:
        Dict with 'content', 'model', 'input_tokens', 'output_tokens', 'total_tokens'.
    """
    client = get_anthropic_client()

    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=max_tokens or settings.ANTHROPIC_MAX_TOKENS,
        system=system_message if system_message else "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )

    content = ""
    for block in response.content:
        if hasattr(block, "text"):
            content += block.text

    return {
        "content": content,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
    }
