"""LLM pipeline service with OpenAI primary and Anthropic fallback."""

from typing import Any

from app.core.logging import get_logger
from app.llm.anthropic_client import call_anthropic
from app.llm.openai_client import call_openai
from app.llm.prompts import (
    get_attribution_report_prompt,
    get_program_reasoning_prompt,
    get_scholarship_reasoning_prompt,
)

logger = get_logger(__name__)


class LLMPipelineService:
    """Handles LLM calls with automatic provider fallback."""

    async def generate_reasoning(
        self,
        user_profile: dict[str, Any],
        entity_data: dict[str, Any],
        score_breakdown: dict[str, float],
        strengths: list[str],
        gaps: list[str],
    ) -> dict[str, Any]:
        """Generate match reasoning via LLM with fallback.

        Returns:
            Dict with 'content', 'model', 'fallback_used', token counts.
        """
        context = {
            "user_profile": user_profile,
            "entity_data": entity_data,
            "score_breakdown": score_breakdown,
            "strengths": strengths,
            "gaps": gaps,
        }

        entity_type = entity_data.get("entity_type", "program")
        if entity_type == "scholarship":
            prompt = get_scholarship_reasoning_prompt(context)
        else:
            prompt = get_program_reasoning_prompt(context)

        try:
            result = await call_openai(
                prompt=prompt,
                system_message="You are an academic advisor helping evaluate student-program fit.",
            )
            result["fallback_used"] = False
            logger.info("llm_reasoning_openai_success", model=result.get("model"))
            return result
        except Exception as openai_exc:  # pylint: disable=broad-exception-caught
            logger.warning("llm_reasoning_openai_failed", error=str(openai_exc))

        try:
            result = await call_anthropic(
                prompt=prompt,
                system_message="You are an academic advisor helping evaluate student-program fit.",
            )
            result["fallback_used"] = True
            logger.info("llm_reasoning_anthropic_fallback_success", model=result.get("model"))
            return result
        except Exception as anthropic_exc:
            logger.error("llm_reasoning_all_providers_failed", error=str(anthropic_exc))
            raise

    async def generate_attribution(
        self,
        match_id: str,
        user_profile: dict[str, Any],
        entity_data: dict[str, Any],
        score_breakdown: dict[str, float],
        match_score: float,
    ) -> dict[str, Any]:
        """Generate a structured attribution report via LLM."""
        context = {
            "user_profile": user_profile,
            "entity_data": entity_data,
            "score_breakdown": score_breakdown,
            "match_score": match_score,
            "match_id": match_id,
        }

        prompt = get_attribution_report_prompt(context)

        try:
            result = await call_openai(
                prompt=prompt,
                system_message="Generate a JSON attribution report for the given match evaluation.",
                response_format="json",
            )
            result["fallback_used"] = False
            return result
        except Exception as openai_exc:  # pylint: disable=broad-exception-caught
            logger.warning("attribution_openai_failed", error=str(openai_exc))

        result = await call_anthropic(
            prompt=prompt,
            system_message="Generate a JSON attribution report for the given match evaluation.",
        )
        result["fallback_used"] = True
        return result
