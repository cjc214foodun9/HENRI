# Phase 8.37 Preregistration — Zone C Neuro-Symbolic Integration

Spec source: HENRI-ANALYSIS-SOTA-BOTTLENECKS-2026 §3.2 ("Populate Zone C
Trajectory Engrams via CEGIS Self-Play" + "Bridge Wave-JEPA Planning with
Zone C Factual Retrieval"). Branch: feat/phase835-analog-traveling-wave-vla
(verified tip 9870fba). Target: remote RTX 5090 + live Zone C
(pgvector 0.6.0 + TimescaleDB 2.29.1, DSN /workspace/zonec_prod.env,
authorized socket path for admin probes).

## Components
- A: CEGIS self-play harvest > 10,000 authorized (o_t, a_t, o_t+1) tuples
  (cegis_self_play_sandbox.py, live arcade, no eval-cache reads).
- B: Streaming ingest of the sealed bank into Zone C
  (phylogenetic_engrams_65536 + zone_c_engrams; deterministic ids,
  idempotent, bounded-batch commits, no DDL — additive rows only).
- C: Default-OFF pgvector retrieval bridge (flag HENRI_ZONEC_BRIDGE=1),
  no production consumer change.

## Pre-registered gates
- G1 HARVEST: bank accepted >= 10,000, all 6 classes >= 1,500,
  manifest digest verified. Kill: < 10,000 -> BLOCKED, do not ingest.
- G2 INGEST: OBSERVED Zone C count of ingested rows >= 10,000
  (phylogenetic_engrams_65536 delta), re-run adds 0 (idempotent).
- G3 RETRIEVAL: HNSW top-1 self-query similarity >= 0.99 and p50 latency
  <= 5 ms (5 reps, OBSERVED on live store).
- G4 REGRESSION: full local suite passes; zero production-consumer
  changes (default-OFF verified by grep); byte-identity preserved.
- G5 VERDICT: sealed verdict doc + commit with run receipts; negatives
  are governance wins (kill = don't wire bridge).

## Constraints
Authorized live data only (provenance in manifest, data_source=authorized).
Zero pre-training invariant: interactive play tuples, not solution labels.
Memory-safe 1-by-1 / bounded-batch streaming ingest (no giant host arrays).
No schema migration (no DDL; reuse existing tables + pkey idempotency).
