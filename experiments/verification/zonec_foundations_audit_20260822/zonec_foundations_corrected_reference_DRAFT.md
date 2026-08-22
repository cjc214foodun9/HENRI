# Zone C Memory Architecture — Corrected Reference (DRAFT FOR APPROVAL)

**Status:** DRAFT — audit-corrected revision of the supplied Academic Foundations text.
**Supersedes:** the unverified design-spec framing of `telemetry_logs`/`wave_checkpoints`/`sagnac_veto_ledger`, EDMD-eigenvalue pruning, 7-day retention, gapfill/locf, and the FE plus-sign.
**Not approved for production docs until user approval.** No behavior changes; read-only audit of 2026-08-22.

---

## 1. Scope and substrate (OBSERVED)

Zone C is a TimescaleDB + pgvector memory governor on the HENRI production stack (timescaledb 2.29.1, pgvector 0.6.0). It stores continuous wave trajectories over a high-dimensional unit hypersphere S^{D-1} with D = 65,536.

- **Full-fidelity waves** (D=65,536, `[num_blocks, 8]`, num_blocks=8192) are stored as **bytea payloads** (`phylogenetic_engrams_65536.engram_wave_bytes`, `boundary_axioms.wave_payload` = 262,144 B/row).
- **Searchable vectors** are 2000-dim Johnson–Lindenstrauss projections (`semantic_index vector(2000)`, `phase_vector vector(2000)`) with HNSW cosine indexes (`phylogenetic_engrams_semantic_hnsw_idx`, `boundary_axioms_semantic_hnsw_idx`).
- **Boundary axioms:** 11 canonical rows (5 crystal + 6 Spelke; dimension 65,536, blocks 8192, rank 16) — OBSERVED.
- **Engram store:** 10,826 engrams (`phylogenetic_engrams_65536` = `zone_c_engrams`); action-family view 10,825 rows; resonant hypersphere telemetry 3,558 rows; `external_outcomes` empty (0 rows).

Retrieval (`zone_c_segment_cache.query_engrams`): HNSW candidate selection over `(id, similarity, age)` with optional `domain_family` filter (CLASS49 Gate 4), then bytea fetch for the top-k only.

## 2. What is actually implemented (OBSERVED / PARTIAL)

| Mechanism | Reality |
|---|---|
| Hypertables | `zone_c_engrams` (7-day chunks), `zone_c_resonant_hypersphere` (1-hour chunks), `external_outcomes` |
| Continuous aggregate | `zone_c_engrams_hourly` — 1h `time_bucket`, AVG/MIN/MAX `sagnac_stress`, COUNT; refreshed hourly by policy |
| Scheduled jobs | Job-history retention (6h), telemetry reporter (24h), cagg refresh (1h) |
| Vector search | pgvector HNSW cosine over 2000-dim projections only |
| Attractor consolidation | `ZoneCAttractorPruner` = greedy cosine clustering at threshold 0.95; in-memory test only; NOT a scheduled prod job |
| Sagnac veto | 0.35 threshold present in code paths; ARC enforcement flag `HENRI_ARC_SAGNAC_VETO` defaults OFF; CLASS48/49 production mean `sagnac_delta` ≈ 0.986–0.988 |

## 3. Not implemented (FALSIFIED — do not claim)

- Tables `telemetry_logs`, `wave_checkpoints`, `sagnac_veto_ledger` do not exist in prod.
- Retention/apoptosis policy (no drop-chunks job; archived 30-day policy is not live and was 30, not 7 days).
- `time_bucket_gapfill` / `locf` — no calls anywhere; database imputation would not by itself guarantee Lie-group phase continuity.
- `first()`/`last()` path-integral energy deltas — no calls.
- EDMD-eigenvalue pruning job: `|λ|≈1` persistent dynamical modes are NOT automatically verified axioms; the live pruner is cosine clustering.
- Hierarchical multi-tier CAGGs (100 Hz→1 Hz→0.01 Hz) and `percentile_agg` — only a single 1h cagg exists.

## 4. Mathematics (corrected)

Variational free energy (canonical form):

```
F = D_KL(q(Ψ) || p(Ψ)) − E_q[log p(o | Ψ)]
```

The likelihood term carries a **minus** sign. The live EFE planner (`efe_planner.py:9-14`) computes

```
EFE(a) = pragmatic_value(a) − epistemic_value(a)
pragmatic_value = Sagnac delta of predicted wave vs boundary axioms
epistemic_value  = information gain / entropy
```

If “pragmatic cost” is defined as negative log-likelihood, then `pragmatic = −E_q[log p(o|Ψ)]` reconciles the two forms; otherwise the displayed plus-sign formula is incorrect. Documentation must state which definition is used.

Chunking bounds B-tree depth in practice but does not by itself establish `O(log N_chunk)` index depth or sub-millisecond insert throughput at 20 kHz — those require measurement. CAGGs + HNSW reduce query work; they do not “eliminate the von Neumann bottleneck.” “Sub-microsecond context inject,” 50 µs proposal budget, and 20 kHz throughput are targets/kill-gates, not observed steady-state properties.

## 5. Representation boundaries (binding)

1. ARC production loop: continuous `[num_blocks, 8]` S^{D-1} UWE (D=65,536) in bytea.
2. pgvector search: only 2000-dim JL projections; never 65,536-dim waves.
3. Flat `Z_256^65536` uint8 rings: coding/REST paths; never stored as pgvector vectors.
4. Do not conflate the 0.35 veto threshold with the ~0.986 telemetry channel unless the calibration relationship is explicitly resolved.

## 6. Relationship to CLASS50

CLASS50 (draft, unexecuted) is NOT amended by this document. Any change to Sagnac semantics, learning mechanism, thresholds, persistence, or retention requires a versioned packet amendment and explicit approval. RT-MCTS remains default-OFF; main untouched.

## 7. Approval

This corrected reference is a DRAFT. Adopt into official documentation only after explicit user approval.
