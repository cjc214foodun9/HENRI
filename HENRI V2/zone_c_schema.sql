-- Canonical HENRI Zone C schema.
--
-- This file is the single schema source for:
--   1. the disposable Docker dev database, and
--   2. the native PostgreSQL production database on Vast.
--
-- Database creation, package installation, and persistence checks are handled
-- by scripts/zone_c_bootstrap.py. This file must run only after the target
-- database and extensions have passed the bootstrap preflight.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS zone_c_schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    description TEXT NOT NULL
);

INSERT INTO zone_c_schema_migrations (version, description)
VALUES (1, 'canonical Zone C storage, telemetry, and versioned subspace artifacts')
ON CONFLICT (version) DO NOTHING;

-- Environment marker. Production is explicit and is never the dev marker.
-- The development Compose override changes this row to 'dev'.
CREATE TABLE IF NOT EXISTS _zonec_environment (
    id SERIAL PRIMARY KEY,
    environment TEXT NOT NULL CHECK (environment IN ('dev', 'prod')),
    seeded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO _zonec_environment (environment)
SELECT 'prod'
WHERE NOT EXISTS (SELECT 1 FROM _zonec_environment);

-- Full wave checkpoints remain bytea. The semantic projection is the only
-- vector indexed by pgvector. Production wave shape is [8192, 8] float32,
-- therefore the expected payload is 262144 bytes.
CREATE TABLE IF NOT EXISTS phylogenetic_engrams_65536 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    environmental_context_hash VARCHAR(255) NOT NULL,
    semantic_index VECTOR(2000) NOT NULL,
    engram_wave_bytes BYTEA NOT NULL,
    CONSTRAINT phylogenetic_wave_payload_nonempty
        CHECK (octet_length(engram_wave_bytes) > 0)
);

CREATE INDEX IF NOT EXISTS phylogenetic_engrams_timestamp_idx
    ON phylogenetic_engrams_65536 (timestamp DESC);

CREATE INDEX IF NOT EXISTS phylogenetic_engrams_context_idx
    ON phylogenetic_engrams_65536 (environmental_context_hash, timestamp DESC);

CREATE INDEX IF NOT EXISTS phylogenetic_engrams_semantic_hnsw_idx
    ON phylogenetic_engrams_65536 USING hnsw (semantic_index vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Stress-attributed engrams. This is a time-series table and is the source of
-- the optional continuous aggregate. The vector dimension matches the live
-- 2000-dimensional semantic projection contract.
CREATE TABLE IF NOT EXISTS zone_c_engrams (
    time TIMESTAMPTZ NOT NULL DEFAULT now(),
    axiom_id UUID NOT NULL,
    domain_tag VARCHAR(128) NOT NULL,
    phase_vector VECTOR(2000) NOT NULL,
    sagnac_stress DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (axiom_id, time)
);

SELECT create_hypertable(
    'zone_c_engrams', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS zone_c_engrams_domain_time_idx
    ON zone_c_engrams (domain_tag, time DESC);

-- Per-step telemetry sink.
CREATE TABLE IF NOT EXISTS zone_c_resonant_hypersphere (
    id UUID NOT NULL,
    domain VARCHAR(64) NOT NULL,
    subdomain VARCHAR(64) NOT NULL,
    concept_key VARCHAR(128) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    real_phases REAL[] NOT NULL,
    imag_phases REAL[] NOT NULL,
    phase_delta REAL NOT NULL,
    sagnac_clearance BOOLEAN NOT NULL,
    PRIMARY KEY (id, recorded_at)
);

SELECT create_hypertable(
    'zone_c_resonant_hypersphere', 'recorded_at',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS zone_c_resonant_domain_time_idx
    ON zone_c_resonant_hypersphere (domain, recorded_at DESC);

-- Low-frequency stress rollup. This summarizes observations; it does not
-- select or prove an invariant axiom.
CREATE MATERIALIZED VIEW IF NOT EXISTS zone_c_engrams_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 hour', time) AS bucket,
    domain_tag,
    AVG(sagnac_stress) AS mean_sagnac_stress,
    MIN(sagnac_stress) AS min_sagnac_stress,
    MAX(sagnac_stress) AS max_sagnac_stress,
    COUNT(*) AS resonance_hits
FROM zone_c_engrams
GROUP BY bucket, domain_tag
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'zone_c_engrams_hourly',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Versioned subspace artifacts. Raw checkpoints are not modified. A basis is
-- preferred over a dense d-by-d projector. No HNSW index is created here until
-- a producer proves a bounded, versioned query representation.
CREATE TABLE IF NOT EXISTS zone_c_subspace_artifacts_v1 (
    artifact_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    domain_tag VARCHAR(128) NOT NULL,
    subspace_type VARCHAR(32) NOT NULL,
    source_operator_id UUID,
    representation VARCHAR(32) NOT NULL,
    ambient_dim INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    basis_bytes BYTEA NOT NULL,
    projector_bytes BYTEA,
    residual_scale DOUBLE PRECISION,
    fit_sample_count BIGINT NOT NULL,
    fit_residual DOUBLE PRECISION NOT NULL,
    validation_status VARCHAR(16) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    provenance_hash VARCHAR(128) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (artifact_id, created_at),
    CHECK (subspace_type IN ('INVARIANT_AXIOM', 'NULL_SPACE_VETO')),
    CHECK (representation IN ('THIN_BASIS', 'PROJECTOR_LOW_RANK')),
    CHECK (ambient_dim > 0),
    CHECK (rank > 0 AND rank <= ambient_dim),
    CHECK (fit_sample_count >= 0),
    CHECK (fit_residual >= 0),
    CHECK (validation_status IN ('CANDIDATE', 'VALIDATED', 'REJECTED'))
);

CREATE INDEX IF NOT EXISTS zone_c_subspace_domain_type_time_idx
    ON zone_c_subspace_artifacts_v1 (domain_tag, subspace_type, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS zone_c_subspace_validated_version_idx
    ON zone_c_subspace_artifacts_v1 (domain_tag, subspace_type, model_version)
    WHERE validation_status = 'VALIDATED';

-- Optional production policies are intentionally not enabled by this base
-- schema. Retention must be selected from observed reuse and backup evidence.
COMMENT ON TABLE zone_c_engrams IS
    'Stress-attributed engrams. Apply retention only after measured reuse and backup review.';
COMMENT ON TABLE zone_c_subspace_artifacts_v1 IS
    'Versioned candidate/validated subspace artifacts; not a task-outcome signal by itself.';
