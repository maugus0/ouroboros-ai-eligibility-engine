"""Pydantic schemas for validating LLM output structure."""

from pydantic import BaseModel, Field


class ProgramReasoningOutput(BaseModel):
    """Expected output from program match reasoning."""

    reasoning: str
    key_strengths: list[str] = Field(default_factory=list)
    key_gaps: list[str] = Field(default_factory=list)
    overall_assessment: str = ""


class ScholarshipReasoningOutput(BaseModel):
    """Expected output from scholarship match reasoning."""

    reasoning: str
    eligibility_summary: str = ""
    competitiveness_assessment: str = ""
    key_strengths: list[str] = Field(default_factory=list)
    key_gaps: list[str] = Field(default_factory=list)


class AttributionReportOutput(BaseModel):
    """Expected output from attribution report generation."""

    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    reasoning: str = ""
    recommendations: list[dict[str, str]] = Field(default_factory=list)
    confidence: str = Field(default="medium")
