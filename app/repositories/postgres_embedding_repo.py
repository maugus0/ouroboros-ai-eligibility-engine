"""Research embeddings CRUD + vector similarity search using raw SQL with asyncpg."""

from typing import Any, Optional

from app.core.logging import get_logger
from app.repositories.postgres_base import PostgresBaseRepository

logger = get_logger(__name__)


class ResearchEmbeddingRepository(PostgresBaseRepository):
    """Repository for research_embeddings table with pgvector operations."""

    async def create(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Insert or update a research embedding and return it."""
        query = """
            INSERT INTO research_embeddings (
                entity_type, entity_id, student_profile_id,
                research_interest_text, embedding, embedding_model, token_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (entity_type, entity_id) DO UPDATE SET
                student_profile_id = EXCLUDED.student_profile_id,
                research_interest_text = EXCLUDED.research_interest_text,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                token_count = EXCLUDED.token_count,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
        """
        return await self.execute_insert_returning(
            query,
            data["entity_type"],
            data["entity_id"],
            data.get("student_profile_id"),
            data["research_interest_text"],
            data["embedding"],
            data.get("embedding_model", "text-embedding-3-small"),
            data.get("token_count"),
        )

    async def get_by_profile_id(self, student_profile_id: str) -> list[dict[str, Any]]:
        """Retrieve all embeddings for a student profile."""
        query = """
            SELECT id, entity_type, entity_id, student_profile_id, research_interest_text,
                   embedding_model, token_count, created_at, updated_at
            FROM research_embeddings
            WHERE student_profile_id = $1
            ORDER BY created_at DESC
        """
        return await self.execute_query(query, student_profile_id)

    async def get_by_entity(self, entity_type: str, entity_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the latest stored embedding for an entity, including the vector."""
        query = """
            SELECT id, entity_type, entity_id, student_profile_id, research_interest_text,
                   embedding, embedding_model, token_count, created_at, updated_at
            FROM research_embeddings
            WHERE entity_type = $1 AND entity_id = $2
            LIMIT 1
        """
        return await self.execute_one(query, entity_type, entity_id)

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        entity_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Find research embeddings similar to the query vector using cosine distance.

        pgvector's <=> operator returns cosine *distance* (1 - similarity).
        We convert to similarity for the result.
        """
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        if entity_type:
            query = """
                SELECT
                    entity_type,
                    entity_id,
                    student_profile_id,
                    research_interest_text,
                    1 - (embedding <=> $1::vector) AS similarity_score
                FROM research_embeddings
                WHERE entity_type = $2
                  AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
            """
            return await self.execute_query(query, embedding_str, entity_type, similarity_threshold, top_k)

        query = """
            SELECT
                entity_type,
                entity_id,
                student_profile_id,
                research_interest_text,
                1 - (embedding <=> $1::vector) AS similarity_score
            FROM research_embeddings
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """
        return await self.execute_query(query, embedding_str, similarity_threshold, top_k)

    async def delete_by_profile_id(self, student_profile_id: str) -> str:
        """Delete all embeddings for a student profile."""
        query = "DELETE FROM research_embeddings WHERE student_profile_id = $1"
        return await self.execute_write(query, student_profile_id)
