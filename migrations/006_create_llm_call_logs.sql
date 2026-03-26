-- Migration 006: LLM Call Logs
-- Audit trail for all LLM API calls

CREATE TABLE IF NOT EXISTS llm_call_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    operation VARCHAR(100) NOT NULL,
    match_id UUID REFERENCES match_results(id) ON DELETE SET NULL,

    llm_provider VARCHAR(20) NOT NULL CHECK (llm_provider IN ('openai', 'anthropic')),
    model_name VARCHAR(100) NOT NULL,

    input_tokens INTEGER,
    output_tokens INTEGER,
    total_cost_usd DECIMAL(10,6),
    latency_ms INTEGER,

    success BOOLEAN NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    trace_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_call_logs_match_id ON llm_call_logs(match_id);
CREATE INDEX IF NOT EXISTS idx_llm_call_logs_trace_id ON llm_call_logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_llm_call_logs_operation ON llm_call_logs(operation);
CREATE INDEX IF NOT EXISTS idx_llm_call_logs_created_at ON llm_call_logs(created_at DESC);
