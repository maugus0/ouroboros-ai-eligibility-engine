"""Pydantic schemas for match evaluation requests and results."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Type of entity being matched against."""

    PROGRAM = "program"
    SCHOLARSHIP = "scholarship"


class ConfidenceLevel(str, Enum):
    """Confidence level for a match score."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""

    user_id: str
    entity_type: EntityType
    entity_id: str
    user_profile: dict[str, Any] = Field(
        ...,
        description="Student profile data (GPA, skills, courses, research, etc.)",
    )
    entity_data: dict[str, Any] = Field(
        ...,
        description="Program or scholarship requirements and details",
    )
    include_attribution: bool = Field(
        default=True,
        description="Whether to generate an attribution report alongside the score",
    )


class BatchEvaluateItem(BaseModel):
    """One entity to evaluate inside a batch request."""

    entity_type: EntityType
    entity_id: str
    entity_data: dict[str, Any] = Field(
        ...,
        description="Program or scholarship requirements and details",
    )
    include_attribution: Optional[bool] = Field(
        default=None,
        description="Optional per-item override for attribution generation",
    )


class BatchEvaluateRequest(BaseModel):
    """Request body for batch evaluation across multiple entities."""

    user_id: str
    user_profile: dict[str, Any] = Field(
        ...,
        description="Student profile data shared across all evaluations in the batch",
    )
    evaluations: list[BatchEvaluateItem] = Field(
        ...,
        min_length=1,
        description="Entities to evaluate for the same user profile",
    )
    include_attribution: bool = Field(
        default=True,
        description="Default attribution behavior applied when an item does not override it",
    )


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of how the match score was computed."""

    gpa: float = Field(default=0.0, ge=0.0, description="GPA component score")
    relevance: float = Field(default=0.0, ge=0.0, description="Field relevance component score")
    prerequisites: float = Field(default=0.0, ge=0.0, description="Prerequisites component score")
    research_alignment: float = Field(default=0.0, ge=0.0, description="Research alignment component score")
    practical_factors: float = Field(default=0.0, ge=0.0, description="Practical factors component score")


class ScholarshipScoreBreakdown(BaseModel):
    """Detailed breakdown for scholarship match scoring."""

    eligibility: float = Field(default=0.0, ge=0.0, description="Eligibility criteria component score")
    preferred_criteria: float = Field(default=0.0, ge=0.0, description="Preferred criteria component score")
    funding_coverage: float = Field(default=0.0, ge=0.0, description="Funding coverage component score")
    competition_estimate: float = Field(default=0.0, ge=0.0, description="Competition estimate component score")


class MatchResult(BaseModel):
    """A single match evaluation result."""

    id: str
    user_id: str
    entity_type: EntityType
    entity_id: str
    match_score: float = Field(..., ge=0.0, le=100.0)
    score_breakdown: dict[str, float]
    confidence_level: ConfidenceLevel
    llm_model_used: Optional[str] = None
    llm_fallback_used: bool = False
    total_processing_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MatchResultSummary(BaseModel):
    """Lightweight match result for list endpoints."""

    id: str
    entity_type: EntityType
    entity_id: str
    match_score: float
    confidence_level: ConfidenceLevel
    created_at: Optional[datetime] = None
