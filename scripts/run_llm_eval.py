"""Run prompt quality evaluation with LLM-generated outputs and LLM-as-judge scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.llm.openai_client import call_openai
from app.services.llm_service import LLMPipelineService

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "llm_eval"
DEFAULT_OUTPUT_PATH = Path("eval_results.json")
JUDGE_SYSTEM_MESSAGE = (
    "You are a strict LLM quality judge for higher-education eligibility reasoning. "
    "Return valid JSON with keys: score, verdict, strengths, issues, summary. "
    "Score on a 0.0 to 5.0 scale."
)


def _load_fixtures(fixture_dir: Path) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for path in sorted(fixture_dir.glob("*.json")):
        fixtures.append(json.loads(path.read_text(encoding="utf-8")))
    return fixtures


def _derive_strengths_and_gaps(fixture: dict[str, Any]) -> tuple[list[str], list[str]]:
    user_profile = fixture.get("user_profile", {})
    entity_data = fixture.get("entity_data", {})
    score_breakdown = fixture.get("score_breakdown", {})

    strengths: list[str] = []
    gaps: list[str] = []

    minimum_gpa = entity_data.get("minimum_gpa")
    gpa = user_profile.get("gpa_normalized")
    if isinstance(gpa, (int, float)) and isinstance(minimum_gpa, (int, float)):
        if gpa >= minimum_gpa:
            strengths.append("The student comfortably meets the GPA requirement.")
        else:
            gaps.append("The student falls below the stated GPA threshold.")

    missing_prerequisites = sorted(
        set(entity_data.get("prerequisites", [])) - set(user_profile.get("completed_courses", []))
    )
    if missing_prerequisites:
        gaps.append(f"Missing prerequisite coverage: {', '.join(missing_prerequisites)}.")
    elif entity_data.get("prerequisites"):
        strengths.append("The student covers all listed prerequisite coursework.")

    if entity_data.get("requires_leadership"):
        if user_profile.get("leadership_experience"):
            strengths.append("Leadership experience aligns with the scholarship expectations.")
        else:
            gaps.append("Leadership evidence is missing for a leadership-focused opportunity.")

    if entity_data.get("eligible_citizenships"):
        citizenship = str(user_profile.get("citizenship", "")).strip()
        if citizenship and citizenship in entity_data["eligible_citizenships"]:
            strengths.append("Citizenship matches the listed eligibility rules.")
        elif citizenship:
            gaps.append("Citizenship does not satisfy the listed hard eligibility criteria.")

    ranked_dimensions = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
    for dimension, value in ranked_dimensions[:2]:
        strengths.append(f"Strong {dimension.replace('_', ' ')} evidence ({value}).")
    for dimension, value in ranked_dimensions[-2:]:
        if value <= 12:
            gaps.append(f"Weaker {dimension.replace('_', ' ')} contribution ({value}).")

    return list(dict.fromkeys(strengths))[:4], list(dict.fromkeys(gaps))[:4]


async def _generate_candidate_output(
    llm_service: LLMPipelineService,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    scenario_type = fixture["scenario_type"]
    if scenario_type in {"program", "scholarship"}:
        strengths, gaps = _derive_strengths_and_gaps(fixture)
        generated = await llm_service.generate_reasoning(
            user_profile=fixture["user_profile"],
            entity_data=fixture["entity_data"],
            score_breakdown=fixture["score_breakdown"],
            strengths=strengths,
            gaps=gaps,
        )
        return {
            "prompt_type": scenario_type,
            "content": generated["content"],
            "provider": "anthropic" if generated.get("fallback_used") else "openai",
            "model": generated.get("model"),
            "input_tokens": generated.get("input_tokens", 0),
            "output_tokens": generated.get("output_tokens", 0),
            "total_tokens": generated.get("total_tokens", 0),
        }

    if scenario_type == "attribution":
        generated = await llm_service.generate_attribution(
            match_id=fixture["case_id"],
            user_profile=fixture["user_profile"],
            entity_data=fixture["entity_data"],
            score_breakdown=fixture["score_breakdown"],
            match_score=float(fixture["match_score"]),
        )
        return {
            "prompt_type": scenario_type,
            "content": json.dumps(generated, ensure_ascii=False, sort_keys=True),
            "provider": generated.get("provider"),
            "model": generated.get("model"),
            "input_tokens": generated.get("input_tokens", 0),
            "output_tokens": generated.get("output_tokens", 0),
            "total_tokens": generated.get("total_tokens", 0),
        }

    raise ValueError(f"Unsupported fixture scenario_type: {scenario_type}")


async def _judge_output(fixture: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    expected = fixture.get("expected", {})
    prompt = json.dumps(
        {
            "description": fixture["description"],
            "scenario_type": fixture["scenario_type"],
            "expected_overall_quality": expected.get("overall_quality"),
            "must_mention": expected.get("must_mention", []),
            "must_not_claim": expected.get("must_not_claim", []),
            "generated_output": candidate["content"],
        },
        ensure_ascii=False,
        indent=2,
    )

    result = await call_openai(
        prompt=(
            "Evaluate the generated output against the expected higher-education eligibility reasoning criteria. "
            "Reward grounded, specific, and non-hallucinatory feedback. Penalize missing required mentions, "
            "forbidden claims, vague reasoning, or incorrect ineligibility statements.\n\n"
            f"{prompt}"
        ),
        system_message=JUDGE_SYSTEM_MESSAGE,
        response_format="json",
        max_tokens=500,
        temperature=0,
    )

    parsed = json.loads(result["content"])
    score = float(parsed.get("score", 0.0))
    parsed["score"] = max(0.0, min(5.0, score))
    parsed["model"] = result.get("model")
    parsed["judge_input_tokens"] = result.get("input_tokens", 0)
    parsed["judge_output_tokens"] = result.get("output_tokens", 0)
    parsed["judge_total_tokens"] = result.get("total_tokens", 0)
    return parsed


async def _run_eval(fixture_dir: Path, baseline_score: float | None) -> dict[str, Any]:
    fixtures = _load_fixtures(fixture_dir)
    llm_service = LLMPipelineService()
    case_results: list[dict[str, Any]] = []

    for fixture in fixtures:
        candidate = await _generate_candidate_output(llm_service, fixture)
        judge = await _judge_output(fixture, candidate)
        case_results.append(
            {
                "case_id": fixture["case_id"],
                "scenario_type": fixture["scenario_type"],
                "description": fixture["description"],
                "candidate": candidate,
                "judge": judge,
            }
        )

    overall_score = round(mean(case["judge"]["score"] for case in case_results), 3)
    threshold = round((baseline_score - 0.5), 3) if baseline_score is not None else None
    passed = threshold is None or overall_score >= threshold

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": settings.OPENAI_MODEL,
        "baseline_score": baseline_score,
        "failure_threshold": threshold,
        "overall_score": overall_score,
        "passed": passed,
        "case_results": case_results,
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM quality evaluation and baseline comparison.")
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--baseline-score", type=float, default=None)
    args = parser.parse_args()

    results = await _run_eval(args.fixture_dir, args.baseline_score)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_score": results["overall_score"],
                "baseline_score": results["baseline_score"],
                "failure_threshold": results["failure_threshold"],
                "passed": results["passed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
