-- Migration 003: Attribution Reports table
-- Stores explainability data for each match

CREATE TABLE IF NOT EXISTS attribution_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES match_results(id) ON DELETE CASCADE,

    strengths JSONB NOT NULL DEFAULT '[]',
    gaps JSONB NOT NULL DEFAULT '[]',
    reasoning TEXT NOT NULL,
    confidence VARCHAR(10) NOT NULL CHECK (confidence IN ('high', 'medium', 'low')),

    recommendations JSONB DEFAULT '[]',

    llm_provider VARCHAR(20),
    llm_model VARCHAR(100),
    prompt_version VARCHAR(50),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_match_attribution UNIQUE (match_id)
);

CREATE INDEX IF NOT EXISTS idx_attribution_match_id ON attribution_reports(match_id);
CREATE INDEX IF NOT EXISTS idx_attribution_confidence ON attribution_reports(confidence);
CREATE INDEX IF NOT EXISTS idx_attribution_strengths ON attribution_reports USING GIN (strengths);
CREATE INDEX IF NOT EXISTS idx_attribution_gaps ON attribution_reports USING GIN (gaps);
