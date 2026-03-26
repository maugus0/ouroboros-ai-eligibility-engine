-- Migration 002: Match Results table
-- Stores program and scholarship match scores

CREATE TABLE IF NOT EXISTS match_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,

    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('program', 'scholarship')),
    entity_id UUID NOT NULL,

    match_score DECIMAL(5,2) NOT NULL CHECK (match_score >= 0 AND match_score <= 100),

    score_breakdown JSONB NOT NULL,

    confidence_level VARCHAR(10) CHECK (confidence_level IN ('high', 'medium', 'low')),
    llm_model_used VARCHAR(100),
    llm_fallback_used BOOLEAN DEFAULT FALSE,
    total_processing_time_ms INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_user_entity UNIQUE (user_id, entity_type, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_match_user_id ON match_results(user_id);
CREATE INDEX IF NOT EXISTS idx_match_entity ON match_results(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_match_score ON match_results(match_score DESC);
CREATE INDEX IF NOT EXISTS idx_match_created_at ON match_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_score_breakdown ON match_results USING GIN (score_breakdown);

CREATE OR REPLACE FUNCTION update_match_results_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_match_results_updated_at ON match_results;
CREATE TRIGGER trigger_update_match_results_updated_at
BEFORE UPDATE ON match_results
FOR EACH ROW
EXECUTE FUNCTION update_match_results_updated_at();
