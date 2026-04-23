"""Prompt quality checks used by tests and CI quality gates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tiktoken

from app.config import settings
from app.utils.prompt_utils import _PROMPTS_DIR, build_prompt_json, build_prompt_text, load_prompt_template

_UNRESOLVED_PLACEHOLDER_PATTERN = re.compile(r"\{\{[^{}]+\}\}")
_DEFAULT_TOKEN_LIMIT = 1600


@dataclass(frozen=True)
class PromptLintResult:
    """Structured result for one prompt template lint run."""

    filename: str
    passed: bool
    issues: list[str]
    json_token_count: int
    text_token_count: int


def list_prompt_files() -> list[str]:
    """Return prompt file names sorted for deterministic linting."""
    return sorted(path.name for path in Path(_PROMPTS_DIR).glob("*.json"))


def sample_prompt_context(filename: str) -> dict[str, Any]:
    """Representative runtime context used to validate built prompt outputs."""
    contexts: dict[str, dict[str, Any]] = {
        "program_reasoning_v1.json": {
            "user_profile": {
                "gpa_normalized": 3.8,
                "major": "Computer Science",
                "technical_skills": ["Python", "Machine Learning", "NLP"],
                "completed_courses": ["Algorithms", "Statistics", "Machine Learning"],
            },
            "entity_data": {
                "entity_type": "program",
                "minimum_gpa": 3.5,
                "keywords": ["Machine Learning", "AI", "NLP"],
                "prerequisites": ["Algorithms", "Statistics"],
            },
            "score_breakdown": {"gpa": 20.0, "relevance": 28.0, "prerequisites": 24.0},
            "strengths": ["Strong GPA", "Relevant coursework"],
            "gaps": ["Limited research output"],
        },
        "scholarship_reasoning_v1.json": {
            "user_profile": {
                "gpa_normalized": 3.85,
                "citizenship": "Indonesia",
                "leadership_experience": True,
                "community_service_hours": 120,
            },
            "entity_data": {
                "entity_type": "scholarship",
                "minimum_gpa": 3.5,
                "eligible_citizenships": ["Indonesia", "Malaysia"],
                "requires_leadership": True,
            },
            "score_breakdown": {"eligibility": 40.0, "preferred_criteria": 27.0},
            "strengths": ["Meets citizenship rule", "Strong leadership profile"],
            "gaps": ["Competition remains high"],
        },
        "attribution_report_v1.json": {
            "match_id": "match-demo-1",
            "user_profile": {
                "gpa_normalized": 3.7,
                "major": "Electrical Engineering",
            },
            "entity_data": {
                "entity_type": "program",
                "minimum_gpa": 3.5,
                "prerequisites": ["Algorithms", "Statistics"],
            },
            "score_breakdown": {"gpa": 18.0, "relevance": 20.0, "prerequisites": 12.0},
            "match_score": 62.0,
        },
        "research_alignment_v1.json": {
            "user_profile": {
                "research_interests": "Trustworthy AI for healthcare diagnostics",
            },
            "entity_data": {
                "research_focus": [
                    "Trustworthy AI",
                    "Healthcare machine learning",
                    "Natural language processing",
                ],
                "faculty_members": ["Prof. Ada", "Prof. Turing"],
            },
        },
    }
    return contexts.get(filename, {})


def _validate_prompt_template_structure(template: dict[str, Any]) -> tuple[list[str], Optional[dict[str, Any]]]:
    issues: list[str] = []
    prompt_template = template.get("prompt_template")
    if not isinstance(prompt_template, dict):
        issues.append("missing `prompt_template` object")
        return issues, None

    base = prompt_template.get("base")
    if not isinstance(base, dict):
        issues.append("missing `prompt_template.base` object")
        return issues, None

    if not isinstance(base.get("agent_identity"), dict):
        issues.append("missing `agent_identity` section")
    if not isinstance(base.get("task_instructions"), dict):
        issues.append("missing `task_instructions` section")

    output_format = base.get("output_format")
    if not isinstance(output_format, dict):
        issues.append("missing `output_format` section")
    else:
        _validate_output_format(output_format, issues)

    constraints = base.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        issues.append("constraints must be a non-empty list")

    return issues, base


def _validate_output_format(output_format: dict[str, Any], issues: list[str]) -> None:
    fmt = output_format.get("format")
    if fmt not in {"json", "text"}:
        issues.append(f"unsupported output format {fmt!r}")
    schema = output_format.get("schema")
    if not isinstance(schema, dict):
        issues.append("missing `output_format.schema` object")


def _collect_placeholder_issues(
    filename: str,
    template: dict[str, Any],
    *,
    token_limit: int,
) -> tuple[list[str], int, int]:
    issues: list[str] = []
    raw_json = json.dumps(template, ensure_ascii=False)
    unresolved = sorted(set(_UNRESOLVED_PLACEHOLDER_PATTERN.findall(raw_json)))
    if unresolved:
        issues.append(f"unresolved placeholders found: {', '.join(unresolved)}")

    built_json, built_text = _build_prompt_variants(filename)
    built_unresolved = sorted(set(_UNRESOLVED_PLACEHOLDER_PATTERN.findall(f"{built_json}\n{built_text}")))
    if built_unresolved:
        issues.append(f"built prompt contains unresolved placeholders: {', '.join(built_unresolved)}")

    json_token_count, text_token_count = _count_variant_tokens(built_json, built_text)
    issues.extend(_token_limit_issues(json_token_count, text_token_count, token_limit))
    return issues, json_token_count, text_token_count


def _build_prompt_variants(filename: str) -> tuple[str, str]:
    context = sample_prompt_context(filename)
    return build_prompt_json(filename, context), build_prompt_text(filename, context)


def _count_variant_tokens(built_json: str, built_text: str) -> tuple[int, int]:
    return count_tokens_for_model(built_json), count_tokens_for_model(built_text)


def _token_limit_issues(json_token_count: int, text_token_count: int, token_limit: int) -> list[str]:
    issues: list[str] = []
    if json_token_count > token_limit:
        issues.append(f"JSON prompt uses {json_token_count} tokens, above limit {token_limit}")
    if text_token_count > token_limit:
        issues.append(f"text prompt uses {text_token_count} tokens, above limit {token_limit}")
    return issues


def lint_prompt_file(filename: str, *, token_limit: Optional[int] = None) -> PromptLintResult:
    """Validate prompt JSON structure, placeholders, and representative token usage."""
    limit = token_limit or _DEFAULT_TOKEN_LIMIT
    template = load_prompt_template(filename)
    structure_issues, base = _validate_prompt_template_structure(template)
    if base is None:
        return PromptLintResult(filename, False, structure_issues, 0, 0)

    placeholder_issues, json_token_count, text_token_count = _collect_placeholder_issues(
        filename,
        template,
        token_limit=limit,
    )
    issues = [*structure_issues, *placeholder_issues]

    return PromptLintResult(
        filename=filename,
        passed=not issues,
        issues=issues,
        json_token_count=json_token_count,
        text_token_count=text_token_count,
    )


def lint_all_prompts(*, token_limit: Optional[int] = None) -> list[PromptLintResult]:
    """Lint every prompt file under the prompts directory."""
    return [lint_prompt_file(filename, token_limit=token_limit) for filename in list_prompt_files()]


def count_tokens_for_model(text: str, *, model: Optional[str] = None) -> int:
    """Count tokens for the configured model, falling back to cl100k_base."""
    model_name = model or settings.OPENAI_MODEL
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
