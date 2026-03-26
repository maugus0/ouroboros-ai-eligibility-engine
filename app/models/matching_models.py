"""Pydantic schemas for match evaluation requests and results."""

from datetime import datetime
from enum import Enum
from typing import Any

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
    llm_model_used: str | None = None
    llm_fallback_used: bool = False
    total_processing_time_ms: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MatchResultSummary(BaseModel):
    """Lightweight match result for list endpoints."""

    id: str
    entity_type: EntityType
    entity_id: str
    match_score: float
    confidence_level: ConfidenceLevel
    created_at: datetime | None = None
