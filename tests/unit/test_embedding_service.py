"""Unit tests for research embedding generation, reuse, and similarity."""

import pytest

from app.services.embedding_service import EmbeddingService


def test_calculate_cosine_similarity_returns_expected_scores():
    identical = EmbeddingService.calculate_cosine_similarity([1.0, 0.0], [1.0, 0.0])
    orthogonal = EmbeddingService.calculate_cosine_similarity([1.0, 0.0], [0.0, 1.0])
    diagonal = EmbeddingService.calculate_cosine_similarity([1.0, 1.0], [1.0, 0.0])

    assert identical == pytest.approx(1.0)
    assert orthogonal == pytest.approx(0.0)
    assert diagonal == pytest.approx(0.707106, rel=1e-5)


@pytest.mark.asyncio
async def test_get_or_create_embedding_reuses_cached_vector(monkeypatch):
    service = EmbeddingService()

    async def fake_get_by_entity(_entity_type: str, _entity_id: str):
        return {
            "entity_type": "student",
            "entity_id": "user-1",
            "research_interest_text": "NLP systems",
            "embedding": "[1.0,0.0,0.0]",
            "embedding_model": "text-embedding-3-small",
        }

    async def fail_generate_embedding(_text: str):
        raise AssertionError("generate_embedding should not be called when cached embedding matches")

    monkeypatch.setattr(service.embedding_repo, "get_by_entity", fake_get_by_entity)
    monkeypatch.setattr(service, "generate_embedding", fail_generate_embedding)

    result = await service.get_or_create_embedding("student", "user-1", "NLP systems")

    assert result["reused"] is True
    assert result["embedding"] == [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_compute_pair_similarity_stores_program_embedding_when_missing(monkeypatch):
    service = EmbeddingService()
    generated_vectors = {
        "Student NLP interests": [1.0, 0.0, 0.0],
        "Program NLP focus": [0.9, 0.1, 0.0],
    }

    async def fake_get_by_entity(_entity_type: str, _entity_id: str):
        return None

    async def fake_generate_embedding(text: str):
        return generated_vectors[text]

    async def fake_create(data: dict):
        return {
            **data,
            "embedding": data["embedding"],
            "embedding_model": data["embedding_model"],
        }

    monkeypatch.setattr(service.embedding_repo, "get_by_entity", fake_get_by_entity)
    monkeypatch.setattr(service, "generate_embedding", fake_generate_embedding)
    monkeypatch.setattr(service.embedding_repo, "create", fake_create)

    result = await service.compute_pair_similarity(
        student_profile_id="student-1",
        student_research_text="Student NLP interests",
        program_id="program-1",
        program_research_text="Program NLP focus",
    )

    assert result["similarity"] > 0.99
    assert result["student_embedding_reused"] is False
    assert result["program_embedding_reused"] is False
