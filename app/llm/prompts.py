"""Prompt loading and building with runtime context injection.

Templates live under ``prompts/`` as versioned JSON. These helpers are
used by in-process code (e.g. ``LLMPipelineService``), not by HTTP clients.
"""

from typing import Any, Optional

from app.utils.prompt_utils import build_prompt_json, build_prompt_text

_VALID_FORMATS: frozenset[str] = frozenset({"json", "text"})


def _require_prompt_format(fmt: str) -> None:
    if fmt not in _VALID_FORMATS:
        raise ValueError(f"Unsupported prompt format {fmt!r}; expected one of {sorted(_VALID_FORMATS)}.")


def get_program_reasoning_prompt(
    context: Optional[dict[str, Any]] = None,
    fmt: str = "json",
) -> str:
    """Build the program reasoning system prompt."""
    _require_prompt_format(fmt)
    if fmt == "text":
        return build_prompt_text("program_reasoning_v1.json", context)
    return build_prompt_json("program_reasoning_v1.json", context)


def get_scholarship_reasoning_prompt(
    context: Optional[dict[str, Any]] = None,
    fmt: str = "json",
) -> str:
    """Build the scholarship reasoning system prompt."""
    _require_prompt_format(fmt)
    if fmt == "text":
        return build_prompt_text("scholarship_reasoning_v1.json", context)
    return build_prompt_json("scholarship_reasoning_v1.json", context)


def get_attribution_report_prompt(
    context: Optional[dict[str, Any]] = None,
    fmt: str = "json",
) -> str:
    """Build the attribution report system prompt."""
    _require_prompt_format(fmt)
    if fmt == "text":
        return build_prompt_text("attribution_report_v1.json", context)
    return build_prompt_json("attribution_report_v1.json", context)


def get_research_alignment_prompt(
    context: Optional[dict[str, Any]] = None,
    fmt: str = "json",
) -> str:
    """Build the research alignment analysis system prompt."""
    _require_prompt_format(fmt)
    if fmt == "text":
        return build_prompt_text("research_alignment_v1.json", context)
    return build_prompt_json("research_alignment_v1.json", context)
