-- Carrier K5 (TZCSM) — typed Zone C semantic memory. ADDITIVE migration.
--
-- Additive-only by design (approval APPROVE_USER_20260904_K5_TZCSM):
--   * canonical zone_c_schema.sql is NOT modified;
--   * boundary_axioms and every pre-existing table/index are untouched;
--   * this file creates five new tables, 14 per-domain partial HNSW indexes,
--     and migration version 2. All DDL uses IF NOT EXISTS.
-- Governing boundary: world-knowledge-boundary.md (zero-pretraining invariant).
-- No corpus row may be written by this migration; ingestion is a separate
-- gated pipeline (K5 spec sections 4-8).

-- Domain taxonomy (14 domains, shared literal with the K5 spec).
-- Per-domain CHECK lists are kept literal to remain reviewable.

CREATE TABLE IF NOT EXISTS domain_source_manifest (
    source_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL CHECK (domain IN (
        'formal_science','natural_science','medicine','engineering','computing',
        'law','economics','humanities','language','arts','education',
        'practical_skills','safety','governance')),
    title TEXT NOT NULL,
    origin TEXT NOT NULL,
    sha256 VARCHAR(64) NOT NULL UNIQUE,
    license TEXT NOT NULL DEFAULT 'UNKNOWN',
    retrieved_utc TIMESTAMPTZ NOT NULL,
    updated_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_revision TEXT
);

CREATE INDEX IF NOT EXISTS domain_source_manifest_domain_idx
    ON domain_source_manifest (domain, retrieved_utc DESC);

CREATE TABLE IF NOT EXISTS corpus_chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES domain_source_manifest(source_id)
        ON DELETE RESTRICT,
    domain TEXT NOT NULL CHECK (domain IN (
        'formal_science','natural_science','medicine','engineering','computing',
        'law','economics','humanities','language','arts','education',
        'practical_skills','safety','governance')),
    chunk_index INTEGER NOT NULL,
    char_span TEXT NOT NULL,
    chunk_sha256 VARCHAR(64) NOT NULL,
    wave_payload BYTEA NOT NULL
        CONSTRAINT corpus_chunk_wave_payload_nonempty
        CHECK (octet_length(wave_payload) > 0),
    proj VECTOR(2000) NOT NULL,
    claim_count INTEGER NOT NULL DEFAULT 0,
    ingested_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'VERIFIED' CHECK (status IN (
        'VERIFIED','VERIFICATION_ABSTAINED','CONTAMINATION_REJECT')),
    UNIQUE (source_id, chunk_index)
);

-- Per-domain partial HNSW indexes (one per domain). Every retrieval query
-- MUST carry the matching domain filter (K5 section 4.7).
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_formal_science_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'formal_science';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_natural_science_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'natural_science';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_medicine_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'medicine';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_engineering_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'engineering';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_computing_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'computing';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_law_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'law';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_economics_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'economics';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_humanities_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'humanities';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_language_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'language';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_arts_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'arts';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_education_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'education';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_practical_skills_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'practical_skills';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_safety_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'safety';
CREATE INDEX IF NOT EXISTS corpus_chunks_proj_governance_idx
    ON corpus_chunks USING hnsw (proj vector_cosine_ops) WHERE domain = 'governance';

CREATE TABLE IF NOT EXISTS world_claims (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id TEXT NOT NULL REFERENCES corpus_chunks(chunk_id)
        ON DELETE RESTRICT,
    domain TEXT NOT NULL CHECK (domain IN (
        'formal_science','natural_science','medicine','engineering','computing',
        'law','economics','humanities','language','arts','education',
        'practical_skills','safety','governance')),
    claim_text_hash VARCHAR(64) NOT NULL,
    claim_type TEXT NOT NULL,
    verification_status TEXT NOT NULL DEFAULT 'VERIFIED' CHECK (
        verification_status IN ('VERIFIED','VERIFICATION_ABSTAINED')),
    evidence_link TEXT,
    sealed_utc TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS world_claims_domain_status_idx
    ON world_claims (domain, verification_status, sealed_utc DESC);

CREATE TABLE IF NOT EXISTS contradiction_ledger (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id_a UUID NOT NULL REFERENCES world_claims(claim_id)
        ON DELETE CASCADE,
    claim_id_b UUID NOT NULL REFERENCES world_claims(claim_id)
        ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN ('CONTRADICTS','SUPPORTS')),
    sealed_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT contradiction_ledger_no_self CHECK (claim_id_a <> claim_id_b)
);

CREATE INDEX IF NOT EXISTS contradiction_ledger_claim_a_idx
    ON contradiction_ledger (claim_id_a);
CREATE INDEX IF NOT EXISTS contradiction_ledger_claim_b_idx
    ON contradiction_ledger (claim_id_b);

INSERT INTO zone_c_schema_migrations (version, description)
VALUES (2, 'Carrier K5 TZCSM: typed Zone C semantic memory (manifests, chunks, claims, contradiction ledger)')
ON CONFLICT (version) DO NOTHING;
