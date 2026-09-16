-- Zone C artifact DAG: parent-child lineage (additive migration, version 4).
--
-- PURPOSE
--   Give Zone C a first-class parent-child lineage graph over research artifacts so
--   a hypothesis, the run that tested it, and the artifacts it produced form an
--   auditable DAG. Before this migration the dev schema had NO table carrying
--   hypothesis / parent-child / program-trace / Pareto structure: the only lineage
--   was the four flat CLASS49 attribution columns (run_id, arm_id, commit_sha,
--   domain_family) added at version 3.
--
-- LIVE SCHEMA EVIDENCE (read from henri_zonec_dev :5434 before writing this file)
--   tables: _zonec_environment, boundary_axioms, contradiction_ledger, corpus_chunks,
--           domain_source_manifest, external_outcomes, phylogenetic_engrams_65536,
--           world_claims, zone_c_action_engrams (view), zone_c_ast_engrams (view),
--           zone_c_engrams, zone_c_engrams_hourly (view),
--           zone_c_resonant_hypersphere, zone_c_schema_migrations,
--           zone_c_subspace_artifacts_v1
--   views that already exist and must NOT be redefined:
--           zone_c_action_engrams, zone_c_ast_engrams, zone_c_engrams_hourly
--   ledger state: versions 1, 2, 3 present (3 = CLASS49 attribution)
--
-- ADDITIVE ONLY. This file does not DROP, RENAME, relax NOT NULL, or backfill
--   anything. Pre-existing rows are untouched; 'legacy_unattributed' remains the
--   honest classification for rows whose lineage was never recorded.
--
-- FAIL-CLOSED AT WRITE TIME
--   The CLASS49 columns on the engram tables carry DEFAULTs so old writers keep
--   working. That default is exactly what this DAG must not rely on: a NEW lineage
--   row that silently inherits 'legacy_unattributed' is an unattributed row wearing
--   a provenance costume. The lineage table therefore declares its four attribution
--   columns NOT NULL with NO default, so an incomplete write fails at INSERT rather
--   than being recorded as attributed. The guard is structural, not conventional.

BEGIN;

-- ---------------------------------------------------------------- 1. lineage table
CREATE TABLE IF NOT EXISTS zone_c_artifact_lineage (
    artifact_id      UUID        NOT NULL DEFAULT gen_random_uuid(),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- CLASS49 attribution, FAIL-CLOSED: NOT NULL with NO default.
    -- A write that omits any of these raises at INSERT time.
    run_id           VARCHAR(64) NOT NULL,
    arm_id           VARCHAR(32) NOT NULL,
    commit_sha       VARCHAR(40) NOT NULL,
    domain_family    VARCHAR(16) NOT NULL,

    -- DAG structure
    parent_artifact_id UUID      NULL REFERENCES zone_c_artifact_lineage(artifact_id),
    root_artifact_id   UUID      NULL,       -- self-reference to the DAG root
    depth              INTEGER   NOT NULL DEFAULT 0,

    -- artifact identity
    artifact_kind    VARCHAR(32)  NOT NULL,  -- hypothesis | run | metric | artifact | verdict
    artifact_uri     TEXT         NULL,      -- repo-relative path or external locator
    content_hash     VARCHAR(64)  NULL,      -- sha256 of the artifact body when on disk
    hypothesis       TEXT         NULL,      -- only set when artifact_kind='hypothesis'
    parent_hypothesis_id UUID     NULL REFERENCES zone_c_artifact_lineage(artifact_id),

    -- Pareto / objective surface. NULL means "not measured", never 0.
    objective_name   VARCHAR(64)  NULL,
    objective_value  DOUBLE PRECISION NULL,
    pareto_rank      INTEGER      NULL,

    evidence_class   VARCHAR(16)  NOT NULL DEFAULT 'UNKNOWN',
    payload          JSONB        NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT zone_c_artifact_lineage_pk PRIMARY KEY (artifact_id),
    CONSTRAINT zone_c_artifact_lineage_evidence_ck
        CHECK (evidence_class IN ('OBSERVED','DERIVED','INFERRED','HYPOTHESIS',
                                  'FALSIFIED','BLOCKED','ENGAGED_WIRING_ONLY','UNKNOWN')),
    CONSTRAINT zone_c_artifact_lineage_kind_ck
        CHECK (artifact_kind IN ('hypothesis','run','metric','artifact','verdict')),
    CONSTRAINT zone_c_artifact_lineage_depth_ck
        CHECK (depth >= 0),
    -- A root is the only row allowed to have no parent, and it must point at itself.
    CONSTRAINT zone_c_artifact_lineage_root_ck
        CHECK (
            (parent_artifact_id IS NULL AND depth = 0)
            OR (parent_artifact_id IS NOT NULL AND depth >= 1)
        )
);

-- ---------------------------------------------------------------- 2. indexes
CREATE INDEX IF NOT EXISTS idx_zcal_parent
    ON zone_c_artifact_lineage(parent_artifact_id);
CREATE INDEX IF NOT EXISTS idx_zcal_root
    ON zone_c_artifact_lineage(root_artifact_id);
CREATE INDEX IF NOT EXISTS idx_zcal_attribution
    ON zone_c_artifact_lineage(run_id, arm_id);
CREATE INDEX IF NOT EXISTS idx_zcal_family_kind
    ON zone_c_artifact_lineage(domain_family, artifact_kind);
CREATE INDEX IF NOT EXISTS idx_zcal_objective
    ON zone_c_artifact_lineage(objective_name, objective_value)
    WHERE objective_value IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_zcal_hypothesis
    ON zone_c_artifact_lineage(parent_hypothesis_id)
    WHERE parent_hypothesis_id IS NOT NULL;

-- ---------------------------------------------------------------- 3. lineage-closure view
-- Depth-1 ancestry only (deterministic, cheap, non-recursive). A recursive CTE is
-- deliberately avoided here: TimescaleDB/PostgreSQL recursive CTEs over a
-- self-referencing table need explicit cycle guards, and a wrong guard silently
-- truncates a DAG. Callers that need full closure walk the parent pointer.
CREATE OR REPLACE VIEW zone_c_artifact_lineage_edges AS
SELECT
    c.artifact_id            AS child_id,
    c.parent_artifact_id     AS parent_id,
    c.root_artifact_id       AS root_id,
    c.depth                  AS depth,
    c.artifact_kind          AS child_kind,
    p.artifact_kind          AS parent_kind,
    c.run_id                 AS child_run_id,
    p.run_id                 AS parent_run_id,
    c.domain_family          AS domain_family,
    c.evidence_class         AS evidence_class
FROM zone_c_artifact_lineage c
LEFT JOIN zone_c_artifact_lineage p
       ON p.artifact_id = c.parent_artifact_id;

COMMENT ON VIEW zone_c_artifact_lineage_edges IS
    'One-hop parent/child projection over zone_c_artifact_lineage. '
    'parent_id IS NULL identifies a DAG root.';

-- ---------------------------------------------------------------- 4. ledger
INSERT INTO zone_c_schema_migrations (version, description)
VALUES (4, 'Zone C artifact DAG: zone_c_artifact_lineage parent-child lineage with '
           'CLASS49 attribution fail-closed (NOT NULL, no default) + edge view')
ON CONFLICT (version) DO NOTHING;

COMMIT;
