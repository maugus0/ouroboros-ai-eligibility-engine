"""Integration tests for the real pgvector-backed embedding flow."""

# pylint: disable=too-many-locals,broad-exception-caught

import uuid

import pytest

from app.config import settings
from app.repositories.db_pool import DatabasePoolConfig, close_pool, create_pool, get_pool
from app.repositories.postgres_embedding_repo import ResearchEmbeddingRepository
from app.services.embedding_service import EmbeddingService


def _vector(*leading_values: float) -> list[float]:
    """Build a 1536-dim vector with deterministic leading values for tests."""
    vector = [0.0] * settings.EMBEDDING_DIMENSION
    for index, value in enumerate(leading_values):
        vector[index] = value
    return vector


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


async def _cleanup_embeddings(
    entity_ids: list[str],
    student_profile_ids: list[str],
) -> None:
    """Delete test rows written during embedding integration checks."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM research_embeddings WHERE entity_id = ANY($1::uuid[])",
            entity_ids,
        )
        await conn.execute(
            "DELETE FROM research_embeddings WHERE student_profile_id = ANY($1::uuid[])",
            student_profile_ids,
        )


async def _cleanup_embeddings_by_text(research_texts: list[str]) -> None:
    """Delete prior test rows with the same research text to keep assertions stable."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM research_embeddings WHERE research_interest_text = ANY($1::text[])",
            research_texts,
        )


@pytest.mark.anyio
async def test_embedding_service_stores_and_queries_real_pgvector(monkeypatch):
    """Store deterministic embeddings and validate pgvector similarity ordering."""

    vectors_by_text = {
        "Natural language processing for multilingual systems": _vector(1.0, 0.0, 0.0),
        "Quantum physics and condensed matter experiments": _vector(0.0, 1.0, 0.0),
        "Deep learning methods for language understanding": _vector(0.96, 0.04, 0.0),
        "NLP for multilingual assistants": _vector(1.0, 0.0, 0.0),
    }

    async def fake_generate_embedding(_self, text: str) -> list[float]:
        return vectors_by_text[text]

    monkeypatch.setattr(EmbeddingService, "generate_embedding", fake_generate_embedding)

    service = EmbeddingService()
    repo = ResearchEmbeddingRepository()

    entity_ids = [str(uuid.uuid4()) for _ in range(4)]
    nlp_profile_id, physics_profile_id, related_nlp_profile_id, nlp_program_id = entity_ids
    student_profile_ids = [nlp_profile_id, physics_profile_id, related_nlp_profile_id]
    research_texts = list(vectors_by_text.keys())[:-1]
    pool_created = False

    try:
        try:
            await _create_test_pool()
            pool_created = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Integration DB unavailable: {exc}")

        await _cleanup_embeddings_by_text(research_texts)

        created_nlp = await service.store_embedding(
            nlp_profile_id,
            "Natural language processing for multilingual systems",
        )
        created_physics = await service.store_embedding(
            physics_profile_id,
            "Quantum physics and condensed matter experiments",
        )
        created_related_nlp = await service.store_embedding(
            related_nlp_profile_id,
            "Deep learning methods for language understanding",
        )

        assert created_nlp is not None
        assert created_physics is not None
        assert created_related_nlp is not None
        assert str(created_nlp["student_profile_id"]) == nlp_profile_id
        assert created_nlp["entity_type"] == "student"
        assert created_nlp["embedding_model"] == settings.OPENAI_EMBEDDING_MODEL
        assert created_nlp["token_count"] > 0

        stored_records = await repo.get_by_profile_id(nlp_profile_id)
        assert len(stored_records) == 1
        assert stored_records[0]["research_interest_text"] == "Natural language processing for multilingual systems"
        assert str(stored_records[0]["entity_id"]) == nlp_profile_id

        created_program = await service.store_program_embedding(
            nlp_program_id,
            "Deep learning methods for language understanding",
        )
        assert created_program is not None
        assert created_program["entity_type"] == "program"
        assert str(created_program["entity_id"]) == nlp_program_id

        pair_similarity = await service.compute_pair_similarity(
            student_profile_id=nlp_profile_id,
            student_research_text="Natural language processing for multilingual systems",
            program_id=nlp_program_id,
            program_research_text="Deep learning methods for language understanding",
        )
        assert pair_similarity["similarity"] > 0.9
        assert pair_similarity["student_embedding_reused"] is True
        assert pair_similarity["program_embedding_reused"] is True

        results = await service.search_similar(
            "NLP for multilingual assistants",
            top_k=3,
            similarity_threshold=0.75,
            entity_type="student",
        )

        returned_ids = [str(item["student_profile_id"]) for item in results]

        assert returned_ids == [nlp_profile_id, related_nlp_profile_id]
        assert physics_profile_id not in returned_ids
        assert float(results[0]["similarity_score"]) == pytest.approx(1.0, abs=1e-6)
        assert float(results[1]["similarity_score"]) > 0.9
        assert all(float(item["similarity_score"]) >= 0.75 for item in results)
    finally:
        if pool_created:
            await _cleanup_embeddings(entity_ids, student_profile_ids)
            await _cleanup_embeddings_by_text(research_texts)
        await close_pool()
