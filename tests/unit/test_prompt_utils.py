"""Tests for prompt template loading and context merging."""

import json

from app.utils.prompt_utils import build_prompt_json, build_prompt_text, load_prompt_template, merge_runtime_context


def test_load_prompt_template():
    template = load_prompt_template("program_reasoning_v1.json")
    assert "prompt_template" in template
    assert "base" in template["prompt_template"]


def test_merge_runtime_context_without_context():
    template = load_prompt_template("program_reasoning_v1.json")
    prompt = merge_runtime_context(template, None)
    assert "runtime_context" not in prompt
    assert "agent_identity" in prompt


def test_merge_runtime_context_with_context():
    template = load_prompt_template("program_reasoning_v1.json")
    context = {"user_profile": {"gpa": 3.5}, "program": {"name": "CS PhD"}}
    prompt = merge_runtime_context(template, context)
    assert "runtime_context" in prompt
    assert prompt["runtime_context"]["user_profile"]["gpa"] == 3.5


def test_build_prompt_json():
    result = build_prompt_json("program_reasoning_v1.json", {"test": True})
    parsed = json.loads(result)
    assert "agent_identity" in parsed
    assert parsed["runtime_context"]["test"] is True


def test_build_prompt_text():
    result = build_prompt_text("program_reasoning_v1.json")
    assert "AGENT IDENTITY" in result
    assert "TASK INSTRUCTIONS" in result
