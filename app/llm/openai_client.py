"""OpenAI client with retry logic."""

from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None


def get_openai_client():
    """Lazy-initialise and return the OpenAI client."""
    global _client  # pylint: disable=global-statement
    if _client is None:
        from openai import AsyncOpenAI  # pylint: disable=import-outside-toplevel

        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def call_openai(
    prompt: str,
    system_message: str = "",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[str] = None,
) -> dict[str, Any]:
    """Call OpenAI API with retry logic.

    Returns:
        Dict with 'content', 'model', 'input_tokens', 'output_tokens', 'total_tokens'.
    """
    client = get_openai_client()

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": settings.OPENAI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens or settings.OPENAI_MAX_TOKENS,
        "temperature": temperature if temperature is not None else settings.OPENAI_TEMPERATURE,
    }

    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    usage = response.usage

    return {
        "content": choice.message.content or "",
        "model": response.model,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
