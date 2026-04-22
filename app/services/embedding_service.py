"""Generate and query research interest embeddings via OpenAI + pgvector."""

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
        embedding = await self.generate_embedding(research_interest_text)

        data = {
            "student_profile_id": student_profile_id,
            "research_interest_text": research_interest_text,
            "embedding": str(embedding),
            "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
            "token_count": len(research_interest_text.split()),
        }

        result = await self.embedding_repo.create(data)
        logger.info(
            "embedding_stored",
            student_profile_id=student_profile_id,
            dimension=len(embedding),
        )
        return result

    async def search_similar(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Find research interests similar to the query text."""
        query_embedding = await self.generate_embedding(query_text)
        return await self.embedding_repo.search_similar(
            query_embedding=query_embedding,
            top_k=top_k or settings.VECTOR_SIMILARITY_TOP_K,
            similarity_threshold=similarity_threshold or settings.VECTOR_SIMILARITY_THRESHOLD,
        )
