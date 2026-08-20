# Phase 8.37 Verdict — Zone C Neuro-Symbolic Integration (SEALED)

**Status: ACCEPT (verified live)**
**Date:** 2026-08-19/20
**Branch:** `feat/phase835-analog-traveling-wave-vla`
**Commits:** `d8efa6c` (ingest + bridge + tests + prereg), `f76c9a0` (DSN `*_dsn` key fix), verdict commit (this file)
**Spec:** HENRI-ANALYSIS-SOTA-BOTTLENECKS-2026 §3.2

## Objective
1. Generate **>10,000 authorized (o_t, a_t, o_{t+1}) trajectory tuples** via CEGIS self-play.
2. **Populate Zone C pgvector** (`phylogenetic_engrams_65536`) with streaming ingest.
3. Provide a **default-OFF retrieval bridge** for Wave-JEPA planning (zero-entropy factual baseplate).

## Evidence (all OBSERVED on live CUDA/Zone C target)

| Gate | Contract | Result |
|---|---|---|
| G1 harvest | ≥10k authorized records, all 6 actions | 10,301 + 1,220 = **11,521** (5 envs: dc22, m0r0, ar25, bp35, g50t) |
| G1 balance | min-support per action | Bank1 [1738,1741,1741,1715,1667,1699]; Supp [210,203,203,202,202,200] — all ≥1,667 / ≥200 |
| G2 ingest | live insert, idempotent | Bank1: 8,178 new / 2,123 dup; Supp: 1,182 new / 38 dup; pilot 66-row re-run: 0 inserted |
| G3 retrieval | HNSW ≤ 5 ms, sim ≥ 0.99 | self-query sim 1.000000, p50 **0.183 ms** |
| G4 regression | local suite | **600 passed / 3 skipped** (7 new 8.37 tests) |
| Gate | unique store ≥ 10,000 | **10,703 unique engrams** (1,343 → 10,703) |

## Artifacts
- `zone_c_engram_ingest.py` — streaming batch ingest; deterministic uuid from sha256(psi‖onehot‖nxt); `ON CONFLICT DO NOTHING RETURNING id`; no [D,D] allocs; commits per 500 rows; refuses non-`authorized` manifest.
- `zone_c_retrieval_bridge.py` — default-OFF (`HENRI_ZONEC_BRIDGE`); fail-closed; HNSW `<=>` top-k via `TimescaleZoneCStore.query_engrams`.
- `tests/unit/test_henri_phase837_zonec_ingest.py` — 7 tests.
- `experiments/verification/arc_phase837_zonec_prereg.md` — pre-registered gates.
- Remote: `/root/henri-837-bank/` (10,301 rec, npz sha `504aa47e…`), `/root/henri-837-bank-supp/` (1,220 rec, npz sha `3757e3d1…`).

## Caveats (reported, not hidden)
1. **Duplicate rate 18.7%** (2,161/11,521): correlated revisits inside the same env produce repeated (o,a,o′) triples — deterministic-id dedup correctly counts only unique tuples.
2. **Env coverage 5/6**: cd82 unavailable (make failed); quota selector converges toward productive envs.
3. **Bridge default-OFF**: no production consumer activates it yet — Wave-JEPA planning wiring is 8.38+ scope. The baseplate exists and is probed, not yet consumed.

## Next
- 8.36 zone_c_env CUDA VRAM harness (≤2.0 ms/step).
- 8.38 ARC blueprint components + bridge consumption wiring.
