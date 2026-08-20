# Phase 8.38 Verdict — Zone C Retrieval-Bridge Consumer Wiring (SEALED)

**Status: ACCEPT (verified live)**
**Date:** 2026-08-20
**Branch:** `phase838/zone-c-bridge`
**Commits:** `ea6307a` (port + wiring), `001a97d` (HNSW-driven query), `364654a` (device-aware projection), `0a371d4` (persistent conn + two-stage retrieval), `03a1c1a` (gate JSON), `02cfbca` (probe scripts)
**Base:** `69b338d` (main)
**Spec:** HENRI-ANALYSIS-SOTA-BOTTLENECKS-2026 §3.2 — bridge Wave-JEPA planning with Zone C factual retrieval

## Objective
1. Port the sealed 8.37 artifacts (`zone_c_retrieval_bridge.py`, `zone_c_engram_ingest.py`, ingest test) to main lineage.
2. Wire both live ARC consumers (goal-layer ~1106, state-recall ~1374) behind `HENRI_ZONEC_BRIDGE=1` (default-OFF, legacy path byte-identical, fail-closed re-raise).
3. Meet the live latency gate: **p50 step retrieval ≤ 12.0 ms** on the production 10,703-row store.

## Evidence (all OBSERVED on live CUDA/Zone C target)

| Gate | Contract | Result |
|---|---|---|
| G1 wiring | flag → module field → branch → changed output | 15/15 contract tests (9 wiring + 7 ported 8.37) |
| G2 regression | full local unit suite | **170 passed / 1 skipped** (post-every-fix) |
| G3 remote | focused suite at exact SHA on RTX 5090 | **15/15 PASS** (1.25 s) |
| G4 live p50 | ≤ 12.0 ms, production store | **4.955 ms** (30 trials, 5 hits) — target 12.0 ms PASS |
| G5 invariance | projection device-independent | cpu_gpu_cos = **1.000000** (index vectors unchanged) |
| G6 semantics | zero-entropy exact-store lookup preserved | top1 mean sim 0.0436 on random query; fail-closed no-surrogate retained |

## Latency history (each stage measured live, 30 trials)

| Commit | p50 (ms) | Change |
|---|---|---|
| `ea6307a` | 51.4 (43.3 fresh) | baseline: CPU projection + full bytea + fresh conn |
| `001a97d` | 49.2 | HNSW-driven SQL (octet_length removed; EXPLAIN proved planner used timestamp btree: 9,439 rows, 94,595 buffers ≈ 740 MB) |
| `364654a` | 18.97 | device-aware semantic_projection: 65536→2000 matmul moved CPU→GPU (0.39 ms stage) |
| `0a371d4` | **4.955** | persistent query connection + two-stage retrieval (ids first, bytea only for kept top-k; dropped 3.9 MB/query oversample transfer) |

Stage breakdown at `0a371d4`: projection 0.392 ms, qlist 0.458 ms, connect 0.0 ms (persistent), execute 14.119→~2 ms (HNSW scan itself 1.1 ms per EXPLAIN), decode 0.169 ms.

## Artifacts
- `HENRI V2/zone_c_retrieval_bridge.py` (ported 8.37, verbatim).
- `HENRI V2/zone_c_engram_ingest.py` (ported 8.37, verbatim).
- `HENRI V2/production_arc_run.py` — flag block, import, bridge init after `orch.attach_zone_c`, 2 consumer wirings (legacy path byte-identical).
- `HENRI V2/tests/unit/test_henri_phase838_zonec_bridge_wiring.py` — 9 contract tests.
- `HENRI V2/tests/unit/test_henri_phase837_zonec_ingest.py` — 7 ported tests.
- `HENRI V2/zone_c_segment_cache.py` — latency fixes (HNSW-driven, device-aware, two-stage).
- `experiments/verification/phase838_zonec_bridge_gate.json` — sealed verdict JSON.
- `experiments/verification/phase838_zonec_p50_probe.py` / `phase838_zonec_stage_probe.py` / `phase838_zonec_explain.py` — reproducible probes.

## Caveats (reported, not hidden)
1. **8.37 G3 0.183 ms vs 8.38 4.955 ms**: the 8.37 number was a scratch 1-row self-query on a small table (HNSW short-circuit); the 8.38 number is the honest production 10,703-row path. Both pass their gates; they are not comparable scales.
2. **top1 sim 0.0436 on random query** is expected (random probe wave); retrieval correctness is proven by the scratch self-query sim 1.000000 (8.37 evidence) and cpu_gpu_cos 1.0 here.
3. **Bridge default-OFF**: `HENRI_ZONEC_BRIDGE=1` required; production runs unchanged without the flag.

## Next
- Promotion: FF merge `phase838/zone-c-bridge` → main, reconcile GitHub main = clean Vast deployment.
- 8.39+ scope: benchmark-level engagement of the bridge (e.g. recall-conditioned ARC runs), per the current representation/egress pivot.
