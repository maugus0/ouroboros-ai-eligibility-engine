"""Integration tests for the matching -> persistence -> attribution flow."""

# pylint: disable=too-many-locals,too-many-statements,broad-exception-caught

import uuid
from typing import Optional

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.repositories.db_pool import DatabasePoolConfig, close_pool, create_pool, get_pool
from app.repositories.postgres_attribution_repo import AttributionReportRepository
from app.repositories.postgres_history_repo import ScoringHistoryRepository
from app.repositories.postgres_match_repo import MatchResultRepository
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMPipelineService


async def _create_test_pool() -> None:
    """Initialise the shared asyncpg pool for integration tests."""
    await create_pool(
        DatabasePoolConfig(
            host=settings.get_db_host(),
            port=settings.get_db_port(),
            database=settings.get_db_name(),
            user=settings.get_db_user(),
            password=settings.get_db_password(),
            min_size=1,
            max_size=5,
            timeout=settings.DB_CONNECTION_TIMEOUT,
        )
    )


async def _cleanup_match_records(user_id: str) -> None:
    """Delete test rows; child rows cascade from match_results."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM match_results WHERE user_id = $1", user_id)


async def _cleanup_pool_resources(user_id: Optional[str], pool_created: bool) -> None:
    """Only run cleanup if the integration DB pool was successfully created."""
    if not pool_created:
        return
    if user_id is not None:
        await _cleanup_match_records(user_id)
    await close_pool()


def _program_payload(
    user_id: str,
    entity_id: str,
    include_attribution: bool = True,
) -> dict:
    """Build a representative program evaluation payload for integration checks."""
    return {
        "user_id": user_id,
        "entity_type": "program",
        "entity_id": entity_id,
        "user_profile": {
            "gpa_normalized": 3.8,
            "major": "Computer Science",
            "technical_skills": ["Python", "Machine Learning", "AI", "NLP"],
            "completed_courses": ["Algorithms", "Machine Learning", "Data Structures"],
            "research_interests": "Natural language processing and multilingual AI systems",
            "preferred_locations": ["Sydney"],
            "budget_usd": 40000,
        },
        "entity_data": {
            "entity_type": "program",
            "minimum_gpa": 3.5,
            "keywords": ["Machine Learning", "AI", "NLP"],
            "prerequisites": ["Algorithms", "Machine Learning"],
            "location": "Sydney",
            "tuition_usd": 35000,
        },
        "include_attribution": include_attribution,
    }


def _scholarship_payload(
    user_id: str,
    entity_id: str,
    include_attribution: bool = True,
) -> dict:
    """Build a representative scholarship evaluation payload for integration checks."""
    return {
        "user_id": user_id,
        "entity_type": "scholarship",
        "entity_id": entity_id,
        "user_profile": {
            "gpa_normalized": 3.8,
            "citizenship": "Singapore",
            "field_of_study": "Computer Science",
            "degree_level": "masters",
            "activities": ["leadership", "community_service"],
            "achievements": [],
            "financial_need_usd": 30000,
        },
        "entity_data": {
            "entity_type": "scholarship",
            "minimum_gpa": 3.5,
            "eligible_citizenships": ["Singapore", "Malaysia"],
            "eligible_fields_of_study": ["Computer Science", "Data Science"],
            "required_degree_level": "masters",
            "preferred_criteria": ["leadership", "community_service", "research"],
            "award_amount_usd": 15000,
            "acceptance_rate": 0.2,
        },
        "include_attribution": include_attribution,
    }


def _research_alignment_result(
    score: float,
    provider: str = "openai",
    model: str = "gpt-research-test",
    fallback_used: bool = False,
) -> dict:
    """Build a structured LLM research-alignment result for tests."""
    return {
        "score": score,
        "similarity": score / 100.0,
        "provider": provider,
        "model": model,
        "fallback_used": fallback_used,
        "alignment_summary": "Test alignment summary.",
        "overlapping_themes": ["nlp"],
        "unique_student_interests": [],
        "recommended_faculty": [],
        "confidence": "high",
    }


@pytest.mark.anyio
async def test_orchestrator_request_persists_match_and_exposes_attribution_report(
    service_token_header,
    monkeypatch,
):
    async def fake_assess_research_alignment(_self, user_profile, entity_data):
        _ = (user_profile, entity_data)
        return _research_alignment_result(88.0, model="gpt-research-openai")

    async def fake_generate_attribution(
        _self,
        match_id,
        user_profile,
        entity_data,
        score_breakdown,
        match_score,
    ):
        _ = (match_id, user_profile, entity_data, score_breakdown, match_score)
        return {
            "reasoning": "Mock reasoning for integration flow.",
            "strengths": ["Strong overall alignment"],
            "gaps": [],
            "recommendations": [],
            "confidence": "high",
            "provider": "openai",
            "model": "mock-openai-model",
            "fallback_used": False,
        }

    monkeypatch.setattr(LLMPipelineService, "assess_research_alignment", fake_assess_research_alignment)
    monkeypatch.setattr(LLMPipelineService, "generate_attribution", fake_generate_attribution)

    user_id = str(uuid.uuid4())
    entity_id = str(uuid.uuid4())

    payload = _program_payload(user_id, entity_id, include_attribution=True)

    match_repo = MatchResultRepository()
    history_repo = ScoringHistoryRepository()
    attribution_repo = AttributionReportRepository()
    pool_created = False

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as integration_client:
            response = await integration_client.post(
                "/matching/evaluate",
                json=payload,
                headers=service_token_header,
            )
            assert response.status_code == 200

            body = response.json()
            assert body["success"] is True
            assert "match_result" in body["data"]
            assert "attribution_report" in body["data"]

            match_result = body["data"]["match_result"]
            attribution_report = body["data"]["attribution_report"]
            match_id = match_result["id"]

            assert match_result["user_id"] == user_id
            assert match_result["entity_type"] == "program"
            assert match_result["entity_id"] == entity_id
            assert match_result["confidence_level"] in {"high", "medium", "low"}
            assert attribution_report["match_id"] == match_id
            assert attribution_report["reasoning"] == "Mock reasoning for integration flow."
            assert attribution_report["llm_provider"] == "openai"
            assert attribution_report["llm_model"] == "mock-openai-model"

            stored_match = await match_repo.get_by_id(match_id)
            assert stored_match is not None
            assert str(stored_match["user_id"]) == user_id
            assert str(stored_match["entity_id"]) == entity_id
            assert stored_match["entity_type"] == "program"
            assert float(stored_match["match_score"]) == pytest.approx(float(match_result["match_score"]))
            assert stored_match["llm_model_used"] == "gpt-research-openai"

            stored_history = await history_repo.get_by_match_id(match_id)
            assert len(stored_history) == 1
            assert str(stored_history[0]["match_id"]) == match_id
            assert float(stored_history[0]["computed_score"]) == pytest.approx(float(match_result["match_score"]))
            assert stored_history[0]["scoring_params"]["metadata"]["research_alignment"]["provider"] == "openai"

            stored_report = await attribution_repo.get_by_match_id(match_id)
            assert stored_report is not None
            assert str(stored_report["match_id"]) == match_id
            assert stored_report["reasoning"] == "Mock reasoning for integration flow."
            assert stored_report["llm_provider"] == "openai"

            report_response = await integration_client.get(
                f"/attribution/report/{match_id}",
                headers=service_token_header,
            )
            assert report_response.status_code == 200
            report_body = report_response.json()
            assert report_body["success"] is True
            assert report_body["data"]["id"] == attribution_report["id"]
            assert report_body["data"]["match_id"] == match_id
            assert report_body["data"]["reasoning"] == "Mock reasoning for integration flow."
    finally:
        await _cleanup_pool_resources(user_id, pool_created)


@pytest.mark.anyio
async def test_orchestrator_request_without_attribution_persists_match_and_supports_queries(
    service_token_header,
    monkeypatch,
):
    async def fake_assess_research_alignment(_self, user_profile, entity_data):
        _ = (user_profile, entity_data)
        return _research_alignment_result(72.0)

    monkeypatch.setattr(LLMPipelineService, "assess_research_alignment", fake_assess_research_alignment)

    user_id = str(uuid.uuid4())
    entity_id = str(uuid.uuid4())
    payload = _program_payload(user_id, entity_id, include_attribution=False)

    match_repo = MatchResultRepository()
    attribution_repo = AttributionReportRepository()
    pool_created = False

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as integration_client:
            response = await integration_client.post(
                "/matching/evaluate",
                json=payload,
                headers=service_token_header,
            )
            assert response.status_code == 200
            body = response.json()
            assert body["success"] is True
            assert "match_result" in body["data"]
            assert "attribution_report" not in body["data"]

            match_id = body["data"]["match_result"]["id"]

            stored_match = await match_repo.get_by_id(match_id)
            assert stored_match is not None
            assert str(stored_match["user_id"]) == user_id

            stored_report = await attribution_repo.get_by_match_id(match_id)
            assert stored_report is None

            list_response = await integration_client.get(
                f"/matching/results/{user_id}",
                params={"entity_type": "program", "page": 1, "page_size": 10},
                headers=service_token_header,
            )
            assert list_response.status_code == 200
            list_body = list_response.json()
            assert list_body["success"] is True
            assert list_body["data"]["total"] >= 1
            assert any(item["id"] == match_id for item in list_body["data"]["items"])

            detail_response = await integration_client.get(
                f"/matching/results/detail/{match_id}",
                headers=service_token_header,
            )
            assert detail_response.status_code == 200
            detail_body = detail_response.json()
            assert detail_body["success"] is True
            assert detail_body["data"]["id"] == match_id

            missing_report_response = await integration_client.get(
                f"/attribution/report/{match_id}",
                headers=service_token_header,
            )
            assert missing_report_response.status_code == 200
            missing_report_body = missing_report_response.json()
            assert missing_report_body["success"] is False
            assert missing_report_body["message"] == "Attribution report not found"
            assert missing_report_body["data"] is None
    finally:
        await _cleanup_pool_resources(user_id, pool_created)


@pytest.mark.anyio
async def test_program_output_baseline_uses_expected_score_and_rule_based_attribution(
    service_token_header,
    monkeypatch,
):
    async def fake_assess_research_alignment(_self, user_profile, entity_data):
        _ = (user_profile, entity_data)
        return _research_alignment_result(80.0)

    async def fail_generate_attribution(
        _self,
        match_id,
        user_profile,
        entity_data,
        score_breakdown,
        match_score,
    ):
        _ = (match_id, user_profile, entity_data, score_breakdown, match_score)
        raise RuntimeError("LLM unavailable for baseline verification")

    monkeypatch.setattr(LLMPipelineService, "assess_research_alignment", fake_assess_research_alignment)
    monkeypatch.setattr(LLMPipelineService, "generate_attribution", fail_generate_attribution)

    user_id = str(uuid.uuid4())
    entity_id = str(uuid.uuid4())
    payload = {
        "user_id": user_id,
        "entity_type": "program",
        "entity_id": entity_id,
        "user_profile": {
            "gpa_normalized": 3.6,
            "major": "Computer Science",
            "technical_skills": ["Python", "Machine Learning", "AI"],
            "completed_courses": ["Algorithms"],
            "research_interests": "Natural language processing",
            "preferred_locations": ["Sydney"],
            "budget_usd": 20000,
        },
        "entity_data": {
            "entity_type": "program",
            "minimum_gpa": 3.5,
            "keywords": ["Machine Learning", "AI", "NLP", "Statistics"],
            "prerequisites": ["Algorithms", "Statistics"],
            "location": "Sydney",
            "tuition_usd": 35000,
        },
        "include_attribution": True,
    }

    expected_breakdown = {
        "gpa": 20.0,
        "relevance": 22.5,
        "prerequisites": 12.5,
        "research_alignment": 12.0,
        "practical_factors": 8.5,
    }
    expected_score = 75.5

    match_repo = MatchResultRepository()
    history_repo = ScoringHistoryRepository()
    attribution_repo = AttributionReportRepository()
    pool_created = False

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as integration_client:
            response = await integration_client.post(
                "/matching/evaluate",
                json=payload,
                headers=service_token_header,
            )
            assert response.status_code == 200
            body = response.json()
            assert body["success"] is True

            match_result = body["data"]["match_result"]
            attribution_report = body["data"]["attribution_report"]
            match_id = match_result["id"]

            assert float(match_result["match_score"]) == pytest.approx(expected_score)
            assert match_result["confidence_level"] == "high"
            for key, expected_value in expected_breakdown.items():
                assert float(match_result["score_breakdown"][key]) == pytest.approx(expected_value)

            assert attribution_report["llm_provider"] == "rule_based"
            assert attribution_report["llm_model"] is None
            assert attribution_report["confidence"] == "high"
            assert attribution_report["strengths"] == [
                "Strong gpa score (20.0)",
                "Strong relevance score (22.5)",
                "Overall strong match (75.5/100)",
            ]
            assert attribution_report["gaps"] == ["Missing prerequisite: Statistics"]
            assert attribution_report["recommendations"] == [{"action": "Complete Statistics", "priority": "high"}]
            assert attribution_report["reasoning"] == (
                "Match score: 75.5/100. "
                "Strengths: Strong gpa score (20.0); Strong relevance score (22.5); "
                "Overall strong match (75.5/100). "
                "Areas for improvement: Missing prerequisite: Statistics."
            )

            stored_match = await match_repo.get_by_id(match_id)
            assert stored_match is not None
            assert float(stored_match["match_score"]) == pytest.approx(expected_score)
            assert stored_match["llm_model_used"] == "gpt-research-test"

            stored_history = await history_repo.get_by_match_id(match_id)
            assert len(stored_history) == 1
            assert float(stored_history[0]["computed_score"]) == pytest.approx(expected_score)
            assert stored_history[0]["scoring_params"]["metadata"]["research_alignment"]["similarity"] == pytest.approx(
                0.8
            )
            assert stored_history[0]["scoring_params"]["metadata"]["research_alignment"]["provider"] == "openai"

            stored_report = await attribution_repo.get_by_match_id(match_id)
            assert stored_report is not None
            assert stored_report["llm_provider"] == "rule_based"
            assert stored_report["reasoning"] == attribution_report["reasoning"]
    finally:
        await _cleanup_pool_resources(user_id, pool_created)


@pytest.mark.anyio
async def test_orchestrator_auth_validation_and_not_found_behaviour(service_token_header):
    missing_user_id = str(uuid.uuid4())
    pool_created = False
    invalid_payload = {
        "user_id": str(uuid.uuid4()),
        "entity_type": "invalid-type",
        "entity_id": str(uuid.uuid4()),
        "user_profile": {},
        "entity_data": {},
        "include_attribution": True,
    }

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as integration_client:
            missing_token = await integration_client.post(
                "/matching/evaluate",
                json=_program_payload(str(uuid.uuid4()), str(uuid.uuid4())),
            )
            assert missing_token.status_code == 401
            assert missing_token.json()["detail"] == "X-Service-Token header required"

            invalid_token = await integration_client.post(
                "/matching/evaluate",
                json=_program_payload(str(uuid.uuid4()), str(uuid.uuid4())),
                headers={"X-Service-Token": "wrong-token"},
            )
            assert invalid_token.status_code == 403
            assert invalid_token.json()["detail"] == "Invalid service token"

            invalid_entity_type = await integration_client.post(
                "/matching/evaluate",
                json=invalid_payload,
                headers=service_token_header,
            )
            assert invalid_entity_type.status_code == 422

            missing_match = await integration_client.get(
                f"/matching/results/detail/{uuid.uuid4()}",
                headers=service_token_header,
            )
            assert missing_match.status_code == 200
            assert missing_match.json() == {
                "success": False,
                "message": "Match result not found",
                "data": None,
            }

            missing_report = await integration_client.get(
                f"/attribution/report/{uuid.uuid4()}",
                headers=service_token_header,
            )
            assert missing_report.status_code == 200
            assert missing_report.json() == {
                "success": False,
                "message": "Attribution report not found",
                "data": None,
            }

            empty_results = await integration_client.get(
                f"/matching/results/{missing_user_id}",
                headers=service_token_header,
            )
            assert empty_results.status_code == 200
            empty_body = empty_results.json()
            assert empty_body["success"] is True
            assert empty_body["data"]["items"] == []
            assert empty_body["data"]["total"] == 0
            assert empty_body["data"]["total_pages"] == 0
    finally:
        await _cleanup_pool_resources(None, pool_created)


@pytest.mark.anyio
async def test_scholarship_output_baseline_uses_expected_score_and_rule_based_attribution(
    service_token_header,
    monkeypatch,
):
    async def fail_generate_attribution(
        _self,
        match_id,
        user_profile,
        entity_data,
        score_breakdown,
        match_score,
    ):
        _ = (match_id, user_profile, entity_data, score_breakdown, match_score)
        raise RuntimeError("LLM unavailable for scholarship baseline verification")

    monkeypatch.setattr(LLMPipelineService, "generate_attribution", fail_generate_attribution)

    user_id = str(uuid.uuid4())
    entity_id = str(uuid.uuid4())
    payload = _scholarship_payload(user_id, entity_id, include_attribution=True)

    expected_breakdown = {
        "eligibility": 40.0,
        "preferred_criteria": 26.0,
        "funding_coverage": 10.0,
        "competition_estimate": 2.0,
    }
    expected_score = 78.0

    match_repo = MatchResultRepository()
    history_repo = ScoringHistoryRepository()
    attribution_repo = AttributionReportRepository()
    pool_created = False

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as integration_client:
            response = await integration_client.post(
                "/matching/evaluate",
                json=payload,
                headers=service_token_header,
            )
            assert response.status_code == 200
            body = response.json()
            assert body["success"] is True

            match_result = body["data"]["match_result"]
            attribution_report = body["data"]["attribution_report"]
            match_id = match_result["id"]

            assert float(match_result["match_score"]) == pytest.approx(expected_score)
            assert match_result["confidence_level"] == "high"
            for key, expected_value in expected_breakdown.items():
                assert float(match_result["score_breakdown"][key]) == pytest.approx(expected_value)

            assert attribution_report["llm_provider"] == "rule_based"
            assert attribution_report["llm_model"] is None
            assert attribution_report["confidence"] == "high"
            assert attribution_report["strengths"] == [
                "Strong eligibility score (40.0)",
                "Strong preferred criteria score (26.0)",
                "Overall strong match (78.0/100)",
            ]
            assert attribution_report["gaps"] == ["Weak competition estimate — needs improvement"]
            assert attribution_report["recommendations"] == [
                {"action": "Address: Weak competition estimate — needs improvement", "priority": "medium"}
            ]
            assert attribution_report["reasoning"] == (
                "Match score: 78.0/100. "
                "Strengths: Strong eligibility score (40.0); Strong preferred criteria score (26.0); "
                "Overall strong match (78.0/100). "
                "Areas for improvement: Weak competition estimate — needs improvement."
            )

            stored_match = await match_repo.get_by_id(match_id)
            assert stored_match is not None
            assert stored_match["entity_type"] == "scholarship"
            assert float(stored_match["match_score"]) == pytest.approx(expected_score)
            assert stored_match["llm_model_used"] is None

            stored_history = await history_repo.get_by_match_id(match_id)
            assert len(stored_history) == 1
            assert float(stored_history[0]["computed_score"]) == pytest.approx(expected_score)
            assert stored_history[0]["scoring_params"]["metadata"]["competition_estimate"][
                "acceptance_rate"
            ] == pytest.approx(0.2)

            stored_report = await attribution_repo.get_by_match_id(match_id)
            assert stored_report is not None
            assert stored_report["llm_provider"] == "rule_based"
            assert stored_report["reasoning"] == attribution_report["reasoning"]
    finally:
        await _cleanup_pool_resources(user_id, pool_created)


@pytest.mark.anyio
async def test_program_evaluation_degrades_gracefully_when_research_similarity_fails(
    service_token_header,
    monkeypatch,
):
    async def failing_assess_research_alignment(_self, user_profile, entity_data):
        _ = (user_profile, entity_data)
        raise RuntimeError("llm alignment failed")

    async def failing_search_similar(_self, query_text: str, top_k=None, similarity_threshold=None):
        _ = (query_text, top_k, similarity_threshold)
        raise RuntimeError("pgvector lookup failed")

    async def fail_generate_attribution(
        _self,
        match_id,
        user_profile,
        entity_data,
        score_breakdown,
        match_score,
    ):
        _ = (match_id, user_profile, entity_data, score_breakdown, match_score)
        raise RuntimeError("LLM unavailable for degradation verification")

    monkeypatch.setattr(LLMPipelineService, "assess_research_alignment", failing_assess_research_alignment)
    monkeypatch.setattr(EmbeddingService, "search_similar", failing_search_similar)
    monkeypatch.setattr(LLMPipelineService, "generate_attribution", fail_generate_attribution)

    user_id = str(uuid.uuid4())
    entity_id = str(uuid.uuid4())
    payload = _program_payload(user_id, entity_id, include_attribution=True)

    match_repo = MatchResultRepository()
    history_repo = ScoringHistoryRepository()
    attribution_repo = AttributionReportRepository()
    pool_created = False

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as integration_client:
            response = await integration_client.post(
                "/matching/evaluate",
                json=payload,
                headers=service_token_header,
            )
            assert response.status_code == 200
            body = response.json()
            assert body["success"] is True

            match_result = body["data"]["match_result"]
            attribution_report = body["data"]["attribution_report"]
            match_id = match_result["id"]

            assert float(match_result["match_score"]) == pytest.approx(85.0)
            assert float(match_result["score_breakdown"]["research_alignment"]) == pytest.approx(0.0)
            assert match_result["confidence_level"] == "high"

            stored_match = await match_repo.get_by_id(match_id)
            assert stored_match is not None
            assert float(stored_match["score_breakdown"]["research_alignment"]) == pytest.approx(0.0)

            stored_history = await history_repo.get_by_match_id(match_id)
            assert len(stored_history) == 1
            assert stored_history[0]["scoring_params"]["metadata"]["research_alignment"]["similarity"] == pytest.approx(
                0.0
            )
            assert stored_history[0]["scoring_params"]["metadata"]["research_alignment"]["provider"] == "unavailable"

            stored_report = await attribution_repo.get_by_match_id(match_id)
            assert stored_report is not None
            assert stored_report["llm_provider"] == "rule_based"
            assert "Weak research alignment" in stored_report["reasoning"]
            assert attribution_report["reasoning"] == stored_report["reasoning"]
    finally:
        await _cleanup_pool_resources(user_id, pool_created)


@pytest.mark.anyio
async def test_results_query_contracts_cover_pagination_filtering_sorting_and_detail(
    service_token_header,
    monkeypatch,
):
    async def fake_assess_research_alignment(_self, user_profile, entity_data):
        _ = entity_data
        mapping = {
            "high-fit research": 0.9,
            "low-fit research": 0.1,
        }
        query_text = user_profile["research_interests"]
        return _research_alignment_result(mapping[query_text] * 100.0)

    monkeypatch.setattr(LLMPipelineService, "assess_research_alignment", fake_assess_research_alignment)

    user_id = str(uuid.uuid4())
    high_program_entity_id = str(uuid.uuid4())
    low_program_entity_id = str(uuid.uuid4())
    scholarship_entity_id = str(uuid.uuid4())
    pool_created = False

    high_program_payload = _program_payload(user_id, high_program_entity_id, include_attribution=False)
    high_program_payload["user_profile"]["research_interests"] = "high-fit research"

    low_program_payload = {
        "user_id": user_id,
        "entity_type": "program",
        "entity_id": low_program_entity_id,
        "user_profile": {
            "gpa_normalized": 3.0,
            "major": "Computer Science",
            "technical_skills": ["Python"],
            "completed_courses": ["Algorithms"],
            "research_interests": "low-fit research",
            "preferred_locations": [],
            "budget_usd": 20000,
        },
        "entity_data": {
            "entity_type": "program",
            "minimum_gpa": 3.5,
            "keywords": ["Machine Learning", "AI", "NLP", "Statistics"],
            "prerequisites": ["Algorithms", "Statistics"],
            "location": "Sydney",
            "tuition_usd": 35000,
        },
        "include_attribution": False,
    }

    scholarship_payload = _scholarship_payload(user_id, scholarship_entity_id, include_attribution=False)

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as integration_client:
            high_program_response = await integration_client.post(
                "/matching/evaluate",
                json=high_program_payload,
                headers=service_token_header,
            )
            low_program_response = await integration_client.post(
                "/matching/evaluate",
                json=low_program_payload,
                headers=service_token_header,
            )
            scholarship_response = await integration_client.post(
                "/matching/evaluate",
                json=scholarship_payload,
                headers=service_token_header,
            )

            assert high_program_response.status_code == 200
            assert low_program_response.status_code == 200
            assert scholarship_response.status_code == 200

            high_program_match = high_program_response.json()["data"]["match_result"]
            low_program_match = low_program_response.json()["data"]["match_result"]
            scholarship_match = scholarship_response.json()["data"]["match_result"]

            assert float(high_program_match["match_score"]) > float(scholarship_match["match_score"])
            assert float(scholarship_match["match_score"]) > float(low_program_match["match_score"])

            filtered_page_1 = await integration_client.get(
                f"/matching/results/{user_id}",
                params={"entity_type": "program", "page": 1, "page_size": 1},
                headers=service_token_header,
            )
            assert filtered_page_1.status_code == 200
            filtered_page_1_body = filtered_page_1.json()
            assert filtered_page_1_body["success"] is True
            assert filtered_page_1_body["data"]["total"] == 2
            assert filtered_page_1_body["data"]["page"] == 1
            assert filtered_page_1_body["data"]["page_size"] == 1
            assert filtered_page_1_body["data"]["total_pages"] == 2
            assert [item["id"] for item in filtered_page_1_body["data"]["items"]] == [high_program_match["id"]]

            filtered_page_2 = await integration_client.get(
                f"/matching/results/{user_id}",
                params={"entity_type": "program", "page": 2, "page_size": 1},
                headers=service_token_header,
            )
            assert filtered_page_2.status_code == 200
            filtered_page_2_body = filtered_page_2.json()
            assert [item["id"] for item in filtered_page_2_body["data"]["items"]] == [low_program_match["id"]]

            all_results = await integration_client.get(
                f"/matching/results/{user_id}",
                params={"page": 1, "page_size": 10},
                headers=service_token_header,
            )
            assert all_results.status_code == 200
            all_results_body = all_results.json()
            assert all_results_body["data"]["total"] == 3
            assert [item["id"] for item in all_results_body["data"]["items"]] == [
                high_program_match["id"],
                scholarship_match["id"],
                low_program_match["id"],
            ]

            scholarship_detail = await integration_client.get(
                f"/matching/results/detail/{scholarship_match['id']}",
                headers=service_token_header,
            )
            assert scholarship_detail.status_code == 200
            scholarship_detail_body = scholarship_detail.json()
            assert scholarship_detail_body["success"] is True
            assert scholarship_detail_body["data"]["id"] == scholarship_match["id"]
            assert scholarship_detail_body["data"]["entity_type"] == "scholarship"
            assert float(scholarship_detail_body["data"]["match_score"]) == pytest.approx(
                float(scholarship_match["match_score"])
            )
    finally:
        await _cleanup_pool_resources(user_id, pool_created)
