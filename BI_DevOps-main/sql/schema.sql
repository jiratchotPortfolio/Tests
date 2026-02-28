-- =============================================================================
-- Sentinel: RAG-Based Customer Insight Engine
-- Database Schema
-- PostgreSQL 15+
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Trigram index for fuzzy text search

-- =============================================================================
-- DIMENSION TABLES
-- =============================================================================

CREATE TABLE agents (
    agent_id        UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_code      VARCHAR(20)     NOT NULL UNIQUE,
    full_name       VARCHAR(120)    NOT NULL,
    team_name       VARCHAR(80),
    hire_date       DATE            NOT NULL,
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE product_categories (
    category_id     SERIAL          PRIMARY KEY,
    category_code   VARCHAR(30)     NOT NULL UNIQUE,
    category_name   VARCHAR(100)    NOT NULL,
    parent_id       INT             REFERENCES product_categories(category_id) ON DELETE RESTRICT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE topics (
    topic_id        SERIAL          PRIMARY KEY,
    topic_name      VARCHAR(100)    NOT NULL UNIQUE,
    category_id     INT             NOT NULL REFERENCES product_categories(category_id) ON DELETE RESTRICT,
    severity_tier   SMALLINT        NOT NULL CHECK (severity_tier BETWEEN 1 AND 5),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- FACT TABLES
-- =============================================================================

CREATE TABLE conversations (
    conversation_id     UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_ref_id     VARCHAR(60)     NOT NULL UNIQUE,  -- CRM system reference
    agent_id            UUID            NOT NULL REFERENCES agents(agent_id) ON DELETE RESTRICT,
    topic_id            INT             REFERENCES topics(topic_id) ON DELETE SET NULL,
    channel             VARCHAR(30)     NOT NULL CHECK (channel IN ('live_chat', 'email', 'phone', 'in_app')),
    language_code       CHAR(5)         NOT NULL DEFAULT 'en',
    started_at          TIMESTAMPTZ     NOT NULL,
    ended_at            TIMESTAMPTZ,
    resolution_status   VARCHAR(30)     NOT NULL CHECK (resolution_status IN ('resolved', 'escalated', 'abandoned', 'pending')),
    csat_score          SMALLINT        CHECK (csat_score BETWEEN 1 AND 5),
    handle_time_secs    INT,
    embedding_id        VARCHAR(100),   -- Foreign reference to vector DB record
    raw_log_checksum    CHAR(64),       -- SHA-256 of pre-processed text; prevents re-embedding
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE conversation_turns (
    turn_id             BIGSERIAL       PRIMARY KEY,
    conversation_id     UUID            NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_index          SMALLINT        NOT NULL,
    speaker_role        VARCHAR(10)     NOT NULL CHECK (speaker_role IN ('customer', 'agent', 'bot')),
    message_text        TEXT            NOT NULL,
    message_tokens      INT,
    sent_at             TIMESTAMPTZ     NOT NULL,
    embedding_chunk_id  VARCHAR(100),   -- Reference to corresponding vector chunk
    search_vector       TSVECTOR,       -- Full-text search representation
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (conversation_id, turn_index)
);

CREATE TABLE escalations (
    escalation_id       BIGSERIAL       PRIMARY KEY,
    conversation_id     UUID            NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    escalated_from      UUID            NOT NULL REFERENCES agents(agent_id),
    escalated_to        UUID            REFERENCES agents(agent_id),
    escalation_reason   TEXT,
    escalated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ,
    resolution_notes    TEXT
);

CREATE TABLE ingestion_batches (
    batch_id            UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_path         TEXT            NOT NULL,
    started_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    records_processed   INT             DEFAULT 0,
    records_failed      INT             DEFAULT 0,
    status              VARCHAR(20)     NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    error_log           JSONB
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- conversations: primary analytical access patterns
CREATE INDEX idx_conv_agent_date     ON conversations (agent_id, started_at DESC);
CREATE INDEX idx_conv_topic_status   ON conversations (topic_id, resolution_status);
CREATE INDEX idx_conv_date_channel   ON conversations (started_at DESC, channel);
CREATE INDEX idx_conv_csat           ON conversations (csat_score) WHERE csat_score IS NOT NULL;
CREATE INDEX idx_conv_pending        ON conversations (created_at DESC) WHERE resolution_status = 'pending';

-- conversation_turns: full-text search
CREATE INDEX idx_turns_fts           ON conversation_turns USING GIN (search_vector);
CREATE INDEX idx_turns_conv          ON conversation_turns (conversation_id, turn_index);
CREATE INDEX idx_turns_text_trgm     ON conversation_turns USING GIN (message_text gin_trgm_ops);

-- agents
CREATE INDEX idx_agents_active       ON agents (team_name) WHERE is_active = TRUE;

-- escalations
CREATE INDEX idx_esc_conv            ON escalations (conversation_id);
CREATE INDEX idx_esc_date            ON escalations (escalated_at DESC);

-- =============================================================================
-- TRIGGER: Auto-update tsvector on insert/update
-- =============================================================================

CREATE OR REPLACE FUNCTION update_turn_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.message_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trig_update_fts
    BEFORE INSERT OR UPDATE OF message_text
    ON conversation_turns
    FOR EACH ROW
    EXECUTE FUNCTION update_turn_search_vector();

-- =============================================================================
-- MATERIALIZED VIEW: Agent performance summary (refreshed on schedule)
-- =============================================================================

CREATE MATERIALIZED VIEW mv_agent_performance AS
SELECT
    a.agent_id,
    a.full_name,
    a.team_name,
    COUNT(c.conversation_id)                                    AS total_conversations,
    AVG(c.csat_score)::NUMERIC(3,2)                            AS avg_csat,
    AVG(c.handle_time_secs)::INT                               AS avg_handle_time_secs,
    SUM(CASE WHEN c.resolution_status = 'resolved' THEN 1 ELSE 0 END)  AS resolved_count,
    SUM(CASE WHEN c.resolution_status = 'escalated' THEN 1 ELSE 0 END) AS escalated_count,
    ROUND(
        SUM(CASE WHEN c.resolution_status = 'resolved' THEN 1 ELSE 0 END)::NUMERIC
        / NULLIF(COUNT(c.conversation_id), 0) * 100, 2
    )                                                           AS resolution_rate_pct,
    DATE_TRUNC('month', MIN(c.started_at))                     AS first_conversation_month
FROM agents a
LEFT JOIN conversations c ON a.agent_id = c.agent_id
GROUP BY a.agent_id, a.full_name, a.team_name
WITH DATA;

CREATE UNIQUE INDEX idx_mv_agent_perf ON mv_agent_performance (agent_id);

-- =============================================================================
-- MATERIALIZED VIEW: Topic trend by week
-- =============================================================================

CREATE MATERIALIZED VIEW mv_topic_weekly_trends AS
SELECT
    t.topic_id,
    t.topic_name,
    pc.category_name,
    DATE_TRUNC('week', c.started_at)::DATE  AS week_start,
    COUNT(*)                                AS conversation_count,
    AVG(c.csat_score)::NUMERIC(3,2)        AS avg_csat,
    AVG(c.handle_time_secs)::INT           AS avg_handle_time_secs
FROM conversations c
JOIN topics t ON c.topic_id = t.topic_id
JOIN product_categories pc ON t.category_id = pc.category_id
GROUP BY t.topic_id, t.topic_name, pc.category_name, DATE_TRUNC('week', c.started_at)
WITH DATA;

CREATE INDEX idx_mv_topic_weekly ON mv_topic_weekly_trends (topic_id, week_start DESC);

-- =============================================================================
-- COMMENT: Refresh strategy
-- Materialized views should be refreshed via a scheduled job (e.g., pg_cron):
--   SELECT cron.schedule('0 * * * *', $$REFRESH MATERIALIZED VIEW CONCURRENTLY mv_agent_performance$$);
--   SELECT cron.schedule('0 * * * *', $$REFRESH MATERIALIZED VIEW CONCURRENTLY mv_topic_weekly_trends$$);
-- =============================================================================
