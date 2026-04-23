"""Tests for prompt quality gates used in CI."""

from app.utils import prompt_lint
from app.utils.prompt_lint import (
    count_tokens_for_model,
    lint_all_prompts,
    lint_prompt_file,
)


def test_all_prompt_templates_pass_lint():
    results = lint_all_prompts()
    failures = {result.filename: result.issues for result in results if not result.passed}
    assert failures == {}


def test_prompt_lint_detects_token_limit_exceeded():
    result = lint_prompt_file("program_reasoning_v1.json", token_limit=20)
    assert result.passed is False
    assert any("above limit 20" in issue for issue in result.issues)


def test_prompt_lint_detects_unresolved_placeholders(monkeypatch):
    monkeypatch.setattr(
        prompt_lint,
        "load_prompt_template",
        lambda _filename: {
            "prompt_template": {
                "base": {
                    "agent_identity": {"role": "Reviewer"},
                    "task_instructions": {"goal": "Check {{entity_name}}"},
                    "constraints": ["Stay factual"],
                    "output_format": {"format": "json", "schema": {"summary": "string"}},
                }
            }
        },
    )
    monkeypatch.setattr(prompt_lint, "build_prompt_json", lambda *_args, **_kwargs: '{"summary":"{{missing}}"}')
    monkeypatch.setattr(prompt_lint, "build_prompt_text", lambda *_args, **_kwargs: "Explain {{missing}}")

    result = prompt_lint.lint_prompt_file("fake_prompt.json", token_limit=200)

    assert result.passed is False
    assert any("unresolved placeholders" in issue for issue in result.issues)


def test_token_count_is_positive_for_prompt_content():
    assert count_tokens_for_model("Prompt lint token counting example") > 0
