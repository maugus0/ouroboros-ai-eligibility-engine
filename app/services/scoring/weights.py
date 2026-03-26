"""Centralised scoring weight constants loaded from settings."""

from app.config import settings


class ProgramScoringWeights:
    """Weights for program match scoring (sum = 100)."""

    GPA = settings.PROGRAM_WEIGHT_GPA
    RELEVANCE = settings.PROGRAM_WEIGHT_RELEVANCE
    PREREQUISITES = settings.PROGRAM_WEIGHT_PREREQUISITES
    RESEARCH_ALIGNMENT = settings.PROGRAM_WEIGHT_RESEARCH
    PRACTICAL_FACTORS = settings.PROGRAM_WEIGHT_PRACTICAL


class ScholarshipScoringWeights:
    """Weights for scholarship match scoring (sum = 100)."""

    ELIGIBILITY = settings.SCHOLARSHIP_WEIGHT_ELIGIBILITY
    PREFERRED_CRITERIA = settings.SCHOLARSHIP_WEIGHT_PREFERRED
    FUNDING_COVERAGE = settings.SCHOLARSHIP_WEIGHT_FUNDING
    COMPETITION_ESTIMATE = settings.SCHOLARSHIP_WEIGHT_COMPETITION
