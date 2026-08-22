# Zone C Academic-Foundations Audit — Claim Ledger

**Date:** 2026-08-22 · **Scope:** read-only audit, prod Zone C (vast-5090, timescaledb 2.29.1, pgvector 0.6.0) + local code at `bae02f9`
**Method:** psql `information_schema` / `timescaledb_information` / `pg_class` probes; grep symbol traces (HENRI V2, excl. `_archive`)

## Verdict summary

| Status | Count | Claims |
|---|---|---|
| OBSERVED | 3 | ZC-03, ZC-16, ZC-17 |
| PARTIAL | 4 | ZC-05, ZC-07, ZC-10, ZC-12 |
| FALSIFIED | 6 | ZC-04, ZC-06, ZC-08, ZC-09, ZC-11, ZC-13 |
| INFERRED | 1 | ZC-14 |
| TARGET_GOAL | 3 | ZC-01, ZC-02, ZC-15 |

The supplied text is a **design specification with mixed fidelity**: the D=65,536 wave substrate and 11 axioms are real; most of the TimescaleDB mechanism claims (retention/apoptosis, pruner-by-eigenvalues, gapfill/locf, first/last, claimed table names) do not exist in the live deployment.

## Ledger

| ID | Claim (condensed) | Status | Evidence / Limitation |
|---|---|---|---|
| ZC-01 | Zone A/B 50 µs proposal budget | TARGET_GOAL | `henri_dual_speed_harness.py:19` K1 kill gate ≤50 µs; one su3 kernel measured ≤50 µs (`experiments/performance/phase818_transducer_cuda_check.py:6`). Full-path latency unmeasured. |
| ZC-02 | 20,000 proposals/sec | TARGET_GOAL | Bandwidth math at 20 kHz (`svd_rank_pcie5_verify.py:137-138`); not measured throughput. |
| ZC-03 | D=65,536 S^{D-1} memory substrate | OBSERVED | `boundary_axioms` dim=65536, blocks=8192, rank=16, payload 262144 B; `phylogenetic_engrams_65536.engram_wave_bytes` bytea. Search columns are 2000-dim projections, not the 65,536-dim waves. |
| ZC-04 | Hypertables telemetry_logs/wave_checkpoints/sagnac_veto_ledger | FALSIFIED | All three ABSENT in prod. Actual: `zone_c_engrams`, `zone_c_resonant_hypersphere`, `external_outcomes`. |
| ZC-05 | Chunked B-Tree O(log N_chunk), sub-ms 20 kHz insert | PARTIAL | Hypertables exist (7-day / 1-hour chunks, `zone_c_schema.sql:72,95`); O(log N)/throughput unmeasured. |
| ZC-06 | Retention apoptosis: 7-day drop | FALSIFIED | No drop-chunks policy in prod; only `policy_job_stat_history_retention`. Archived `_archive` script had 30-day (not 7) — not live. |
| ZC-07 | Hierarchical CAGGs 100 Hz→1 Hz→0.01 Hz + percentile_agg | PARTIAL | One cagg: `zone_c_engrams_hourly` (1h bucket, AVG/MIN/MAX, COUNT). No percentile_agg in code. |
| ZC-08 | time_bucket_gapfill + locf phase continuity | FALSIFIED | No calls in live code; cagg def has neither. |
| ZC-09 | first()/last() path-integral deltas | FALSIFIED | No calls in live code. |
| ZC-10 | pgvector l2/cosine/hamming over Z_256^D + complex waves | PARTIAL | pgvector 0.6.0; cosine HNSW `1 - (semantic_index <=> q)` on `vector(2000)` (`zone_c_segment_cache.py:288`); no l2/hamming calls; searches 2000-dim JL projections only. |
| ZC-11 | add_job pruner, EDMD eigenvalues → axiom crystallization | FALSIFIED | `ZoneCAttractorPruner` exists but does greedy cosine consolidation (threshold 0.95) — no eigenvalue logic, no job in prod, only in-memory unit test. |
| ZC-12 | Sagnac veto ≥0.35 → heat + Zone C rejection | PARTIAL | 0.35 threshold widely present (e.g. `efe_planner.py:902/1595`, `production_arc_run.py:630`) but ARC flag defaults OFF (`production_arc_run.py:200`) and CLASS48/49 ran at mean ~0.986 without shutdown → calibration conflict, must not be conflated. |
| ZC-13 | F = D_KL + E_q[log p] (plus sign) | FALSIFIED (as displayed) | Canonical FE uses MINUS. Code: `efe_planner.py:9-14` EFE = pragmatic − epistemic; pragmatic = Sagnac delta vs axioms. Define pragmatic as −E[log p] explicitly or correct the sign. |
| ZC-14 | Eliminates von Neumann bottleneck | INFERRED | CAGG+HNSW exist; no latency measurement; does not eliminate the bottleneck. |
| ZC-15 | Sub-microsecond context inject | TARGET_GOAL | Only archived comment (`_archive/v1_runner/phylogenetic_memory.py:35`). No live mechanism. |
| ZC-16 | 11 canonical boundary axioms | OBSERVED | count=11, dim 65536/8192/rank16. |
| ZC-17 | Engram store counts | OBSERVED | 10,826 / 10,826 / action view 10,825 / resonant 3,558 / external_outcomes **0**. |

## Representation boundary (must be preserved)

- ARC production loop consumes continuous `[num_blocks, 8]` S^{D-1} UWE (D=65,536).
- pgvector search operates ONLY on 2000-dim `semantic_index`/`phase_vector` JL projections (HNSW cosine).
- Flat `Z_256^65536` uint8 phase rings are used in coding/REST paths, never stored as pgvector vectors.
- Do not claim pgvector natively searches 65,536-dim qFHRR waves.

## Action items

1. Correct FE sign in docs; define pragmatic as negative log-likelihood.
2. Resolve 0.35-veto vs ~0.986 production channel calibration conflict before any new Sagnac-gated packet.
3. CLASS50 untouched: no silent amendment; any change requires a versioned packet amendment + explicit approval; RT-MCTS default-OFF; main untouched.
4. Gate-4 non-vacuity fixture test before promotion (standing).
