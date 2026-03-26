-- Migration 005: Research Embeddings table
-- Vector storage for semantic research alignment matching

CREATE TABLE IF NOT EXISTS research_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    student_profile_id UUID NOT NULL,
    research_interest_text TEXT NOT NULL,

    embedding vector(1536) NOT NULL,

    embedding_model VARCHAR(100) NOT NULL DEFAULT 'text-embedding-3-small',
    token_count INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_research_embeddings_vector
ON research_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_research_embeddings_profile ON research_embeddings(student_profile_id);
CREATE INDEX IF NOT EXISTS idx_research_embeddings_created_at ON research_embeddings(created_at DESC);

CREATE OR REPLACE FUNCTION update_research_embeddings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_research_embeddings_updated_at ON research_embeddings;
CREATE TRIGGER trigger_update_research_embeddings_updated_at
BEFORE UPDATE ON research_embeddings
FOR EACH ROW
EXECUTE FUNCTION update_research_embeddings_updated_at();
