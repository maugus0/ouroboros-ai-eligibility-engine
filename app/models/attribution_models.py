"""Pydantic schemas for attribution / explainability reports."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.matching_models import ConfidenceLevel


class Recommendation(BaseModel):
    """A single actionable recommendation for the student."""

    action: str
    priority: str = Field(default="medium", description="high, medium, or low")


class AttributionReport(BaseModel):
    """Full explainability report for a match evaluation."""

    id: str
    match_id: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    recommendations: list[Recommendation] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    prompt_version: Optional[str] = None
    created_at: Optional[datetime] = None


class AttributionRequest(BaseModel):
    """Internal request for generating an attribution report."""

    match_id: str
    user_profile: dict
    entity_data: dict
    score_breakdown: dict[str, float]
    match_score: float
