-- HENRI world-knowledge projection, schema v1.
-- Additive only: never mix factual knowledge with task outcomes or axioms.
-- Raw source text is intentionally not stored in Zone C.

BEGIN;

CREATE TABLE IF NOT EXISTS zone_c_world_knowledge (
    id UUID PRIMARY KEY,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    observed_at TIMESTAMPTZ NOT NULL,
    source_id VARCHAR(512) NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    chunk_sha256 CHAR(64) NOT NULL,
    claim_id VARCHAR(512) NOT NULL,
    claim_sha256 CHAR(64) NOT NULL,
    contradiction_group VARCHAR(512) NOT NULL,
    domain_family VARCHAR(128) NOT NULL,
    model_id VARCHAR(512) NOT NULL,
    model_revision VARCHAR(512) NOT NULL,
    encoder_version VARCHAR(256) NOT NULL,
    evidence_status VARCHAR(16) NOT NULL DEFAULT 'VERIFIED',
    semantic_index VECTOR(2000) NOT NULL,
    wave_payload BYTEA NOT NULL,
    CONSTRAINT world_knowledge_sha_source CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT world_knowledge_sha_chunk CHECK (chunk_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT world_knowledge_sha_claim CHECK (claim_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT world_knowledge_group_nonempty CHECK (length(contradiction_group) > 0),
    CONSTRAINT world_knowledge_status CHECK (evidence_status IN ('VERIFIED', 'PROVISIONAL', 'REJECTED')),
    CONSTRAINT world_knowledge_wave_size CHECK (octet_length(wave_payload) = 262144),
    CONSTRAINT world_knowledge_source_chunk_unique UNIQUE (source_id, chunk_sha256)
);

CREATE INDEX IF NOT EXISTS world_knowledge_domain_time_idx
    ON zone_c_world_knowledge (domain_family, observed_at DESC);

CREATE INDEX IF NOT EXISTS world_knowledge_claim_group_idx
    ON zone_c_world_knowledge (contradiction_group, claim_sha256);

CREATE INDEX IF NOT EXISTS world_knowledge_semantic_hnsw_idx
    ON zone_c_world_knowledge USING hnsw (semantic_index vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE zone_c_world_knowledge IS
    'Provenance-controlled factual wave projection. No raw source text. Query requires domain isolation and contradiction checks.';

COMMIT;
