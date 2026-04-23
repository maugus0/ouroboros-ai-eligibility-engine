"""Sanity checks for LLM evaluation golden fixtures."""

import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm_eval"


def test_llm_eval_fixture_count_is_between_three_and_five():
    fixture_files = sorted(FIXTURE_DIR.glob("*.json"))
    assert 3 <= len(fixture_files) <= 5


def test_llm_eval_fixtures_have_required_fields():
    for path in FIXTURE_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["case_id"]
        assert payload["scenario_type"] in {"program", "scholarship", "attribution"}
        assert isinstance(payload["user_profile"], dict)
        assert isinstance(payload["entity_data"], dict)
        assert isinstance(payload["expected"], dict)
