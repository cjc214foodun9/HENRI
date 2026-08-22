-- CLASS49 Attribution Migration (HENRI-PACKET-CLASS49-ATTRIBUTION-SAGNAC-2026)
-- Applied to dev Docker (henri-zonec-dev) first, then production (Vast 47411800).
-- Non-destructive: additive columns with deterministic backfill; existing
-- records preserved. CLASS49 F1/F2 fixes: attribution columns on ALL physical
-- storage tables (including the recall table phylogenetic_engrams_65536),
-- canonical domain_family column, and dual-namespace views over live tags.

-- 1. zone_c_engrams (stress rollup; domain_tag already exists)
-- FIX F6 (packet bug): arm_id VARCHAR(16) cannot hold DEFAULT
-- 'legacy_unattributed' (19 chars) -> widened to VARCHAR(32).
ALTER TABLE zone_c_engrams
    ADD COLUMN IF NOT EXISTS run_id VARCHAR(64) DEFAULT 'legacy_unattributed',
    ADD COLUMN IF NOT EXISTS arm_id VARCHAR(32) DEFAULT 'legacy_unattributed',
    ADD COLUMN IF NOT EXISTS commit_sha VARCHAR(40) DEFAULT 'untracked',
    ADD COLUMN IF NOT EXISTS domain_family VARCHAR(16) DEFAULT 'general';

-- 2. phylogenetic_engrams_65536 (recall table -- CLASS49 F1 fix)
ALTER TABLE phylogenetic_engrams_65536
    ADD COLUMN IF NOT EXISTS run_id VARCHAR(64) DEFAULT 'legacy_unattributed',
    ADD COLUMN IF NOT EXISTS arm_id VARCHAR(32) DEFAULT 'legacy_unattributed',
    ADD COLUMN IF NOT EXISTS commit_sha VARCHAR(40) DEFAULT 'untracked',
    ADD COLUMN IF NOT EXISTS domain_family VARCHAR(16) DEFAULT 'general';

-- 3. zone_c_resonant_hypersphere (telemetry hypertable)
ALTER TABLE zone_c_resonant_hypersphere
    ADD COLUMN IF NOT EXISTS run_id VARCHAR(64) DEFAULT 'legacy_unattributed',
    ADD COLUMN IF NOT EXISTS arm_id VARCHAR(32) DEFAULT 'legacy_unattributed',
    ADD COLUMN IF NOT EXISTS commit_sha VARCHAR(40) DEFAULT 'untracked',
    ADD COLUMN IF NOT EXISTS domain_family VARCHAR(16) DEFAULT 'general';

-- Indexes
CREATE INDEX IF NOT EXISTS idx_zone_c_engrams_domain ON zone_c_engrams(domain_tag);
CREATE INDEX IF NOT EXISTS idx_zone_c_engrams_attribution ON zone_c_engrams(run_id, arm_id);
CREATE INDEX IF NOT EXISTS idx_phylo_engrams_family ON phylogenetic_engrams_65536(domain_family);

-- Backfill domain_family deterministically from existing domain tags
-- (live vocabulary: arc3/{env}, {env}:ACTION{n}, arc3/{env}/field_channel_*)
UPDATE zone_c_engrams
   SET domain_family = CASE
         WHEN domain_tag LIKE 'arc3%' OR domain_tag LIKE '%:ACTION%'
              OR domain_tag LIKE '%/field_channel%' THEN 'action'
         ELSE 'general' END
 WHERE domain_family = 'general';

UPDATE phylogenetic_engrams_65536
   SET domain_family = CASE
         WHEN environmental_context_hash LIKE 'arc3%'
              OR environmental_context_hash LIKE '%:ACTION%'
              OR environmental_context_hash LIKE '%/field_channel%' THEN 'action'
         ELSE 'general' END
 WHERE domain_family = 'general';

-- Dual-namespace views (spec alignment; canonical families, non-vacuous)
CREATE OR REPLACE VIEW zone_c_ast_engrams AS
  SELECT * FROM zone_c_engrams
  WHERE LOWER(domain_family) IN ('ast','code','text','math','language','symbolic');

CREATE OR REPLACE VIEW zone_c_action_engrams AS
  SELECT * FROM zone_c_engrams
  WHERE LOWER(domain_family) IN ('action','grid','ode','control','spatial','exteroceptive');
