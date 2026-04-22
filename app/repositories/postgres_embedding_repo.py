"""Research embeddings CRUD + vector similarity search using raw SQL with asyncpg."""

from typing import Any, Optional

from app.core.logging import get_logger
from app.repositories.postgres_base import PostgresBaseRepository

logger = get_logger(__name__)


class ResearchEmbeddingRepository(PostgresBaseRepository):
    """Repository for research_embeddings table with pgvector operations."""

    async def create(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Insert a new research embedding and return it."""
        query = """
            INSERT INTO research_embeddings (
                student_profile_id, research_interest_text,
                embedding, embedding_model, token_count
            )
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
        """
        return await self.execute_insert_returning(
            query,
            data["student_profile_id"],
            data["research_interest_text"],
            data["embedding"],
            data.get("embedding_model", "text-embedding-3-small"),
            data.get("token_count"),
        )

    async def get_by_profile_id(self, student_profile_id: str) -> list[dict[str, Any]]:
        """Retrieve all embeddings for a student profile."""
        query = """
            SELECT id, student_profile_id, research_interest_text,
                   embedding_model, token_count, created_at, updated_at
            FROM research_embeddings
            WHERE student_profile_id = $1
            ORDER BY created_at DESC
        """
        return await self.execute_query(query, student_profile_id)

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Find research embeddings similar to the query vector using cosine distance.

        pgvector's <=> operator returns cosine *distance* (1 - similarity).
        We convert to similarity for the result.
        """
        query = """
            SELECT
                student_profile_id,
                research_interest_text,
                1 - (embedding <=> $1::vector) AS similarity_score
            FROM research_embeddings
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        return await self.execute_query(query, embedding_str, similarity_threshold, top_k)

    async def delete_by_profile_id(self, student_profile_id: str) -> str:
        """Delete all embeddings for a student profile."""
        query = "DELETE FROM research_embeddings WHERE student_profile_id = $1"
        return await self.execute_write(query, student_profile_id)
