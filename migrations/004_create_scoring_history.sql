-- Migration 004: Scoring History table
-- Audit trail for all scoring operations

CREATE TABLE IF NOT EXISTS scoring_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES match_results(id) ON DELETE CASCADE,

    scoring_params JSONB NOT NULL,
    computed_score DECIMAL(5,2) NOT NULL,
    computation_time_ms INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scoring_history_match_id ON scoring_history(match_id);
CREATE INDEX IF NOT EXISTS idx_scoring_history_created_at ON scoring_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scoring_params ON scoring_history USING GIN (scoring_params);
