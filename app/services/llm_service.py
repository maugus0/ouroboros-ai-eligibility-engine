"""LLM pipeline service with OpenAI primary and Anthropic fallback."""

import json
from typing import Any

from app.core.logging import get_logger
from app.llm.anthropic_client import call_anthropic
from app.llm.openai_client import call_openai
from app.llm.prompts import (
    get_attribution_report_prompt,
    get_program_reasoning_prompt,
    get_research_alignment_prompt,
    get_scholarship_reasoning_prompt,
)

logger = get_logger(__name__)


class LLMPipelineService:
    """Handles LLM calls with automatic provider fallback."""

    async def assess_research_alignment(
        self,
        user_profile: dict[str, Any],
        entity_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Score research alignment with an LLM and return structured metadata."""
        context = {
            "user_profile": user_profile,
            "entity_data": entity_data,
        }
        prompt = get_research_alignment_prompt(context)
        system_message = (
            "You are an academic research alignment analyst. "
            "Return valid JSON with a score from 0 to 100 plus concise evidence."
        )

        try:
            result = await call_openai(
                prompt=prompt,
                system_message=system_message,
                response_format="json",
            )
            parsed = self._parse_research_alignment_response(result["content"])
            parsed["provider"] = "openai"
            parsed["model"] = result.get("model")
            parsed["fallback_used"] = False
            logger.info("research_alignment_openai_success", model=result.get("model"))
            return parsed
        except Exception as openai_exc:  # pylint: disable=broad-exception-caught
            logger.warning("research_alignment_openai_failed", error=str(openai_exc))

        result = await call_anthropic(
            prompt=prompt,
            system_message=system_message,
            max_tokens=400,
        )
        parsed = self._parse_research_alignment_response(result["content"])
        parsed["provider"] = "anthropic"
        parsed["model"] = result.get("model")
        parsed["fallback_used"] = True
        logger.info("research_alignment_anthropic_fallback_success", model=result.get("model"))
        return parsed

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

    @staticmethod
    def _parse_research_alignment_response(content: str) -> dict[str, Any]:
        """Parse and validate structured research-alignment output from the LLM."""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Research alignment response was not valid JSON") from exc

        raw_score = parsed.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Research alignment score must be numeric, got {raw_score!r}") from exc

        parsed["score"] = max(0.0, min(100.0, score))
        parsed["alignment_summary"] = str(parsed.get("alignment_summary", "")).strip()
        parsed["overlapping_themes"] = [str(item) for item in parsed.get("overlapping_themes", [])]
        parsed["unique_student_interests"] = [str(item) for item in parsed.get("unique_student_interests", [])]
        parsed["recommended_faculty"] = [str(item) for item in parsed.get("recommended_faculty", [])]
        parsed["confidence"] = str(parsed.get("confidence", "medium")).lower()
        return parsed
