"""Pydantic schemas for research embeddings and vector search."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ResearchEmbedding(BaseModel):
    """A stored research interest embedding."""

    id: str
    entity_type: Literal["student", "program"]
    entity_id: str
    student_profile_id: Optional[str] = None
    research_interest_text: str
    embedding_model: str = "text-embedding-3-small"
    token_count: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class EmbeddingCreateRequest(BaseModel):
    """Request to create a research embedding."""

    entity_type: Literal["student", "program"] = "student"
    entity_id: str
    research_interest_text: str = Field(
        ...,
        min_length=10,
        description="Research interest text to embed (min 10 chars)",
    )


class VectorSearchRequest(BaseModel):
    """Request for vector similarity search."""

    query_text: str = Field(..., min_length=5, description="Text to find similar research interests for")
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class VectorSearchResult(BaseModel):
    """A single result from vector similarity search."""

    student_profile_id: str
    research_interest_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
