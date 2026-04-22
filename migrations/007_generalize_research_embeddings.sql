-- Migration 007: Generalise research embeddings to support students and programs

ALTER TABLE research_embeddings
ADD COLUMN IF NOT EXISTS entity_type VARCHAR(20);

ALTER TABLE research_embeddings
ADD COLUMN IF NOT EXISTS entity_id UUID;

UPDATE research_embeddings
SET entity_type = COALESCE(entity_type, 'student'),
    entity_id = COALESCE(entity_id, student_profile_id)
WHERE entity_type IS NULL
   OR entity_id IS NULL;

ALTER TABLE research_embeddings
ALTER COLUMN entity_type SET NOT NULL;

ALTER TABLE research_embeddings
ALTER COLUMN entity_id SET NOT NULL;

ALTER TABLE research_embeddings
ALTER COLUMN student_profile_id DROP NOT NULL;

WITH ranked_embeddings AS (
    SELECT
        ctid,
        ROW_NUMBER() OVER (
            PARTITION BY entity_type, entity_id
            ORDER BY updated_at DESC NULLS LAST, ctid DESC
        ) AS row_num
    FROM research_embeddings
)
DELETE FROM research_embeddings re
USING ranked_embeddings ranked
WHERE re.ctid = ranked.ctid
  AND ranked.row_num > 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_research_embeddings_entity_unique
ON research_embeddings(entity_type, entity_id);

CREATE INDEX IF NOT EXISTS idx_research_embeddings_entity_lookup
ON research_embeddings(entity_type, entity_id, updated_at DESC);
