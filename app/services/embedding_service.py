"""Generate and query research interest embeddings via OpenAI + pgvector."""

import math
from typing import Any, Optional

from app.config import settings
from app.core.logging import get_logger
from app.repositories.postgres_embedding_repo import ResearchEmbeddingRepository

logger = get_logger(__name__)


class EmbeddingService:
    """Manages research interest embeddings — generation and similarity search."""

    def __init__(self):
        self.embedding_repo = ResearchEmbeddingRepository()
        self._client = None

    def _get_openai_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # pylint: disable=import-outside-toplevel

            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding for the given text."""
        client = self._get_openai_client()
        response = await client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    async def store_embedding(
        self,
        student_profile_id: str,
        research_interest_text: str,
    ) -> Optional[dict[str, Any]]:
        """Generate and store a research interest embedding."""
        return await self.store_student_embedding(student_profile_id, research_interest_text)

    async def store_student_embedding(
        self,
        student_profile_id: str,
        research_interest_text: str,
    ) -> Optional[dict[str, Any]]:
        """Generate and store a student research embedding."""
        return await self._store_entity_embedding(
            entity_type="student",
            entity_id=student_profile_id,
            research_text=research_interest_text,
        )

    async def store_program_embedding(
        self,
        program_id: str,
        research_focus_text: str,
    ) -> Optional[dict[str, Any]]:
        """Generate and store a program research embedding."""
        return await self._store_entity_embedding(
            entity_type="program",
            entity_id=program_id,
            research_text=research_focus_text,
        )

    async def get_or_create_embedding(
        self,
        entity_type: str,
        entity_id: str,
        research_text: str,
    ) -> dict[str, Any]:
        """Return a cached embedding when possible, otherwise generate and persist it."""
        existing = await self.embedding_repo.get_by_entity(entity_type, entity_id)
        if existing and existing.get("research_interest_text") == research_text:
            embedding = self._coerce_embedding(existing.get("embedding"))
            if embedding:
                return {
                    "record": existing,
                    "embedding": embedding,
                    "reused": True,
                }

        stored = await self._store_entity_embedding(entity_type, entity_id, research_text)
        return {
            "record": stored,
            "embedding": self._coerce_embedding(stored.get("embedding") if stored else None),
            "reused": False,
        }

    async def compute_pair_similarity(
        self,
        student_profile_id: str,
        student_research_text: str,
        program_id: str,
        program_research_text: str,
    ) -> dict[str, Any]:
        """Generate or reuse embeddings for a student-program pair and compute cosine similarity."""
        student_embedding_result = await self.get_or_create_embedding(
            entity_type="student",
            entity_id=student_profile_id,
            research_text=student_research_text,
        )
        program_embedding_result = await self.get_or_create_embedding(
            entity_type="program",
            entity_id=program_id,
            research_text=program_research_text,
        )

        student_embedding = student_embedding_result["embedding"]
        program_embedding = program_embedding_result["embedding"]
        similarity = self.calculate_cosine_similarity(student_embedding, program_embedding)

        return {
            "similarity": similarity,
            "student_embedding_reused": student_embedding_result["reused"],
            "program_embedding_reused": program_embedding_result["reused"],
            "student_embedding_model": (student_embedding_result["record"] or {}).get("embedding_model"),
            "program_embedding_model": (program_embedding_result["record"] or {}).get("embedding_model"),
        }

    async def search_similar(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        entity_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Find research interests similar to the query text."""
        query_embedding = await self.generate_embedding(query_text)
        return await self.embedding_repo.search_similar(
            query_embedding=query_embedding,
            top_k=top_k or settings.VECTOR_SIMILARITY_TOP_K,
            similarity_threshold=similarity_threshold or settings.VECTOR_SIMILARITY_THRESHOLD,
            entity_type=entity_type,
        )

    async def _store_entity_embedding(
        self,
        entity_type: str,
        entity_id: str,
        research_text: str,
    ) -> Optional[dict[str, Any]]:
        """Generate and persist an embedding for a student or program entity."""
        embedding = await self.generate_embedding(research_text)
        data = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "student_profile_id": entity_id if entity_type == "student" else None,
            "research_interest_text": research_text,
            "embedding": self._serialize_embedding(embedding),
            "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
            "token_count": len(research_text.split()),
        }

        result = await self.embedding_repo.create(data)
        logger.info(
            "embedding_stored",
            entity_type=entity_type,
            entity_id=entity_id,
            dimension=len(embedding),
        )
        return result

    @staticmethod
    def calculate_cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vector_a or not vector_b or len(vector_a) != len(vector_b):
            return 0.0

        dot_product = sum(left * right for left, right in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(value * value for value in vector_a))
        magnitude_b = math.sqrt(sum(value * value for value in vector_b))
        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        similarity = dot_product / (magnitude_a * magnitude_b)
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> str:
        """Convert an embedding list into pgvector string format."""
        return "[" + ",".join(str(value) for value in embedding) + "]"

    @classmethod
    def _coerce_embedding(cls, raw_embedding: Any) -> list[float]:
        """Normalize stored embedding values into a list of floats."""
        if raw_embedding is None:
            return []
        if isinstance(raw_embedding, list):
            return [float(value) for value in raw_embedding]
        if isinstance(raw_embedding, tuple):
            return [float(value) for value in raw_embedding]

        raw_text = str(raw_embedding).strip()
        if raw_text.startswith("[") and raw_text.endswith("]"):
            raw_text = raw_text[1:-1]
        if not raw_text:
            return []
        return [float(value.strip()) for value in raw_text.split(",") if value.strip()]
