# Phase 2 — Zone C Dual-Subspace Migration Design (Action vs Lexical)

Document Identifier: **HENRI-MIGRATION-ZONEC-DUAL-SUBSPACE-V1**
Status: **DESIGN-ONLY** — no production DDL executed. All DDL/data movement below is marked **REQUIRES_APPROVAL**.
Mirror of roadmap `HENRI_Production_Readiness_Roadmap.md` Phase 2 (M3: "Partition Zone C schema — zero cross-domain retrieval").

## 1. Problem statement

Live catalog probe (OBSERVED 2026-08-20, read-only, DSN never printed):

| Relation | Columns (summary) | Indexes | Rows |
|---|---|---|---|
| `zone_c_engrams` | time timestamptz, axiom_id uuid, domain_tag varchar, phase_vector USER-DEFINED, sagnac_stress float8 | pkey, time_idx, domain_time_idx | 10,703 |
| `phylogenetic_engrams_65536` | id uuid, timestamp, environmental_context_hash, semantic_index USER-DEFINED, engram_wave_bytes bytea | pkey, timestamp_idx, context_idx, semantic_hnsw_idx | 10,703 |
| `boundary_axioms` | axiom_id text, axiom_kind, source, source_commit, validity_scope, dimension, num_blocks, rank, projection_seed, semantic_index, wave_payload bytea, semantic_projection jsonb, residual_scale, confidence, created_at | pkey, kind_idx, semantic_hnsw_idx | 11 |
| `zone_c_engrams_hourly` | (hourly rollup) | — | — |
| `zone_c_resonant_hypersphere` | (resonance ledger) | — | — |
| `zone_c_schema_migrations` | (migration ledger — reuse, do not duplicate) | — | — |
| `zone_c_subspace_artifacts_v1` | (artifact ledger) | — | — |
| `qfhrr_state_events` | **absent** — created on-demand by `qfhrr_readout_ledger` | — | — |
| `_zonec_environment` | env marker | — | — |

`EXT` check aborted after the missing-relation error (transaction aborted); pgvector presence is INFERRED from `semantic_index USER-DEFINED` + `semantic_hnsw_idx` on two tables (HNSW requires pgvector ≥ 0.5).

Failure mode addressed (roadmap Lens A.2): a single flat `zone_c_engrams` stores spatial/action vectors (ARC) and lexical/AST phase vectors (code) under one `domain_tag` column, allowing cross-domain similarity contamination. Category-theoretic requirement: two sub-categories, `C_action` (o_t, a_t, o_t+1 transitions) and `C_lexical` (AST phase-geometric hypervectors), isolated at the table + index level.

Representation constraint (representation-core-audit, must be honored): ARC = continuous UWE `[num_blocks, 8]` S^{D-1} (stored as the existing USER-DEFINED `phase_vector` type); coding/AST = flat `[65536]` uint8 phase ring (IDF-weighted qFHRR). These are two incompatible wave families — the design keeps them in separate tables with separate storage types.

## 2. Target schema (additive, versioned)

### `zone_c_action_engrams` — schema `henri.zonec-action-engrams.v1`
| Column | Type | Constraint |
|---|---|---|
| action_id | uuid | PK default gen_random_uuid() |
| time | timestamptz | NOT NULL default now() |
| env_key | text | NOT NULL (ARC env id, e.g. lf52) |
| episode_id | uuid | NULL |
| step_index | int | NULL |
| action_code | int | NOT NULL (GameAction enum index) |
| action_payload | jsonb | NULL — full `(GameAction, data)` contract |
| observation_hash | bytea | NULL — frame-hash receipt |
| state_phase | USER-DEFINED | NULL — matches existing `zone_c_engrams.phase_vector` type |
| semantic_index | USER-DEFINED | NULL — pgvector column for HNSW |
| sagnac_stress | float8 | NULL |
| provenance | jsonb | NOT NULL default '{}' — {source_commit, schema_id, run_id} |

Indexes: `zone_c_action_engrams_pkey` (PK); `action_time_idx` (time); `action_env_time_idx` (env_key, time); `action_semantic_hnsw_idx` (semantic_index vector_cosine_ops).

### `zone_c_ast_engrams` — schema `henri.zonec-ast-engrams.v1`
| Column | Type | Constraint |
|---|---|---|
| ast_id | uuid | PK default gen_random_uuid() |
| time | timestamptz | NOT NULL default now() |
| source | text | NOT NULL (MBPP | HumanEval | ...) |
| task_id | text | NULL (e.g. HumanEval/23) |
| grammar_key | text | NOT NULL — canonical AST signature |
| phase_ring | bytea | NOT NULL — 65,536 uint8 bytes (IDF-weighted qFHRR) |
| semantic_index | USER-DEFINED | NULL — pgvector column for HNSW |
| corpus_sha256 | char(64) | NULL — canonical corpus digest (MBPP `ccf64cea…`) |
| idf_freqs | jsonb | NULL — node-type IDF frequencies (reproducibility) |
| provenance | jsonb | NOT NULL default '{}' |

Indexes: `zone_c_ast_engrams_pkey` (PK); `ast_time_idx` (time); `ast_source_task_idx` (source, task_id); `ast_semantic_hnsw_idx` (semantic_index vector_cosine_ops).

### `zone_c_engram_metadata` — shared immutable provenance (schema `henri.zonec-metadata.v1`)
| Column | Type | Constraint |
|---|---|---|
| engram_id | uuid | PK |
| domain | text | NOT NULL CHECK (domain IN ('action','ast','axiom','phylogenetic')) |
| source_commit | text | NULL |
| schema_id | text | NOT NULL |
| run_id | uuid | NULL |
| digest_sha256 | char(64) | NULL |
| created_at | timestamptz | NOT NULL default now() |

Indexes: `metadata_domain_schema_idx` (domain, schema_id); `metadata_commit_idx` (source_commit).

Constraints: dimension guards are enforced at ingest with typed errors (65536 uint8 for AST ring; `num_blocks*8` for ARC UWE) — never in the DB. No hard FK from metadata to action/ast tables (keeps rollback simple); `engram_id` is a soft link.

## 3. Index rationale

Query predicates are: (domain filter) + (time range) + (semantic ANN). Composite btree for exact keys (`env_key, time` / `source, task_id`) + HNSW `vector_cosine_ops` for similarity. No full scans on the hot path. HNSW build cost is trivial at current scale (10,703 rows); tune `m`/`ef_construction` only if row counts grow past ~10⁷.

## 4. Zero-downtime sequence (all steps additive)

1. **Catalog + caller audit** — read-only (this design's evidence; callers: `zone_c_epistemic_axiom_harness.py`, `ingest_mbpp_codebook.py`, ARC ingress).
2. **Create new tables + indexes** — additive DDL; legacy tables untouched. **REQUIRES_APPROVAL.**
3. **Dual-read shadow validation** — run identical queries against legacy and new tables; assert schema/type parity.
4. **Chunked backfill with receipts** — `INSERT … SELECT` domain-filtered from `zone_c_engrams` in chunks of ≤ 1,000 rows; per-chunk row count + SHA-256 receipt.
5. **Reconciliation** — row-count equality, per-column digest equality, vector-shape check (65536 dims), domain distribution match.
6. **Default-OFF dual-write** — ingest paths write legacy + new under `HENRI_ZONE_C_DUAL_WRITE=1`; assert no divergence over a bounded window.
7. **Canary read cutover** — point the AST codebook ingest consumer at `zone_c_ast_engrams`; validate similarity queries.
8. **Full read cutover** — all consumers read new tables; flag default-ON after approval.
9. **Rollback window** — legacy tables retained 30 days; rollback = disable new reads/writes, resume legacy reads. No drops before window expiry + approval.

## 5. Rollback

Every step is reversible: toggling the read/write flag returns consumers to untouched legacy tables; new tables are dropped only after the rollback window expires and only with approval. No destructive DDL at any point.

## 6. SQL drafts (NOT APPLIED — **REQUIRES_APPROVAL**)

```sql
-- DRAFT ONLY. Do not execute on Zone C production without explicit approval.
-- Admin path: sudo -u postgres python (psycopg via /var/run/postgresql socket).

CREATE TABLE IF NOT EXISTS zone_c_action_engrams (
    action_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    time            timestamptz NOT NULL DEFAULT now(),
    env_key         text NOT NULL,
    episode_id      uuid,
    step_index      int,
    action_code     int NOT NULL,
    action_payload  jsonb,
    observation_hash bytea,
    state_phase     USER-DEFINED,
    semantic_index  USER-DEFINED,
    sagnac_stress   float8,
    provenance      jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS action_time_idx ON zone_c_action_engrams (time);
CREATE INDEX IF NOT EXISTS action_env_time_idx ON zone_c_action_engrams (env_key, time);
CREATE INDEX IF NOT EXISTS action_semantic_hnsw_idx ON zone_c_action_engrams
    USING hnsw (semantic_index vector_cosine_ops);

CREATE TABLE IF NOT EXISTS zone_c_ast_engrams (
    ast_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    time            timestamptz NOT NULL DEFAULT now(),
    source          text NOT NULL,
    task_id         text,
    grammar_key     text NOT NULL,
    phase_ring      bytea NOT NULL,
    semantic_index  USER-DEFINED,
    corpus_sha256   char(64),
    idf_freqs       jsonb,
    provenance      jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ast_time_idx ON zone_c_ast_engrams (time);
CREATE INDEX IF NOT EXISTS ast_source_task_idx ON zone_c_ast_engrams (source, task_id);
CREATE INDEX IF NOT EXISTS ast_semantic_hnsw_idx ON zone_c_ast_engrams
    USING hnsw (semantic_index vector_cosine_ops);

CREATE TABLE IF NOT EXISTS zone_c_engram_metadata (
    engram_id       uuid PRIMARY KEY,
    domain          text NOT NULL CHECK (domain IN ('action','ast','axiom','phylogenetic')),
    source_commit   text,
    schema_id       text NOT NULL,
    run_id          uuid,
    digest_sha256   char(64),
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS metadata_domain_schema_idx ON zone_c_engram_metadata (domain, schema_id);
CREATE INDEX IF NOT EXISTS metadata_commit_idx ON zone_c_engram_metadata (source_commit);
```

## 7. Open risks

- `qfhrr_state_events` is created on-demand; the migration must not assume it exists or collide with new names (checked: no collision).
- USER-DEFINED `phase_vector` semantics must be confirmed against the writer before backfill (two wave families per representation-core-audit).
- AST `phase_ring`→`semantic_index` cast cost at ingest: ~256 KB/row — trivial for the 974-item codebook scale.
- HNSW parameter tuning deferred until >10⁷ rows.

## 8. Approval gates

| Gate | Criterion | Approver |
|---|---|---|
| A | Design review (this doc) | User |
| B | Shadow validation pass (step 3) | User |
| C | Reconciliation pass (step 5) | User |
| D | Canary pass (step 7) | User |
| E | Full cutover (step 8) + rollback window start | User |

Each executed step is recorded in `zone_c_schema_migrations` with its event hash. DDL is executed by the admin path only after the corresponding gate approval.
