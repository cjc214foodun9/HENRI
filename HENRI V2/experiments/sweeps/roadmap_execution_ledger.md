# Project HENRI V2 Strategic R&D Roadmap — Execution Ledger

Document: `Project HENRI V2 Strategic R&D Roadmap.pdf`
(local `C:/tmp/p814_roadmap.pdf`, SHA-256 `0ca9f7a17141f3012d8d6ade1f7b53f4e1159c33f6dcebda4c894f10d59c22df`, 263 lines)
Branch base per roadmap §4: tip @ `9f1c207` (sealed 8.12 tip = 8.11+8.12 machinery).

## Roadmap item → status mapping (audit 2026-08-16, worktree `C:/tmp/henri-814-wt`)

| ID | Roadmap item | Cited file(s) | Audit result | Status |
|---|---|---|---|---|
| R8.13.1 | Amplitude ingress encoder (A(x,y)·e^{jΘ}) | `o_vsa_ingress_tokenizer.py --mode amplitude_ingress_test` | File exists; **0 argparse args** — phantom CLI (family #8); mechanism EXECUTED as Phase 8.13, SEALED KILL `11668df` (color cos 1.0, shared 0.6124) | EXECUTED-KILL |
| R8.13.2 | Low-rank coupled transition (V W† + R_block, rank 64, QR) | `efe_planner.py` | `LowRankCoupledTransition` IS production default @70–190 (real-valued V,W); complex V,W ∈ ℂ^D×r is the 8.14 wiring target; G2 gate (L_trans < 0.10 after 30 steps) is satisfiable by the 8.11 complex transition (1e-7) | EXISTS + 8.14 carries the complex variant |
| R8.13.3 | Retroactive RESET-penalty ν engine (k=5 window, anisotropic Langevin injection) | `production_arc_run.py` | RESET valence handling EXISTS @878–929 (−1.0 null-action penalty, legit-RESET 0.0). Sliding k=5 window + anisotropic η_d injection into policy weights: **NEW behavioral change to a load-bearing production file** | DEFERRED — requires its own pre-registration + approval; NOT part of 8.14 |
| R8.14 | Zone C hash-chained provenance ledger | `scripts/agentic_event_store.py`, `sync_timescaledb_telemetry.py` | `agentic_event_store.py` EXISTS at repo ROOT (not scripts/); `sync_timescaledb_telemetry.py` MISSING; `zone_c_wave_ledger` hypertable ABSENT; hash-chain machinery EXISTS in `henri_audit.py` | NEW — executed as 8.14-ledger G4 (1,000-update hash-chain verification) |
| R8.15 | Holographic egress head (ℂ^D → 2048 → |V|, GELU+LN, L_obstruct < 1e-4) | `henri_decoder.py`, `functor_flow.py` | `henri_decoder.py` EXISTS (real D→2048→32000 head); `functor_flow.py` MISSING — real file `henri_functor_flow.py`; complex-input egress head is NEW | PRE-REGISTERED — Phase 8.15 design written, implementation awaits approval |
| R8.16 | Triton phase-similarity LUT kernel (≤50 µs) | `qfhrr_kernels.py`, `gpu_verification_suite.py` | `qfhrr_kernels.py` EXISTS (inspect for existing kernels); `gpu_verification_suite.py` MISSING — phantom CLI family #10 | PRE-REGISTERED — Phase 8.16 design written, implementation awaits approval |

## Deviations (pre-registered)

- D1: Phantom CLIs replaced with real artifacts (contract tests + dedicated runners) — family #8/#9/#10 confirmations.
- D2: R8.13.1 NOT re-executed (already sealed kill `11668df`; re-running a sealed falsification wastes GPU) — cross-referenced.
- D3: R8.13.3 deferred (load-bearing production change; requires approval-gated phase).
- D4: R8.14 sync script does not exist → the hash-chain verification targets the REAL `agentic_event_store.py` + `henri_audit.py` machinery; a `zone_c_wave_ledger` TimescaleDB write path is a separate production deployment (needs DSN + migration approval) — this phase proves the hash-chain contract (G4) at the store level.
- D5: R8.15/R8.16 pre-registered only in this pass (design + gates); implementation in subsequent approved phases.
- D6: Branch `feat/complex-boundary-wiring` @ `9f1c207` (roadmap-mandated base). `main` untouched @ `2218ec4`.
- D7: Gate thresholds: roadmap G1 (<0.05 distinct) adopted for 8.14 G1; roadmap G6 (ARC score > 0.0) stays BLOCKED_NO_DEMONSTRATIONS (standing; G4 in 8.14 runner asserts the block state, never fabricates).
- D8: Roadmap G5 (Triton ≤50 µs) measured in 8.16 phase, not 8.14 (Triton kernel does not exist yet).

## 2026-08-16 — Phase 8.15 PDF ingestion (HENRI-SPEC-2026-08-PHASE8.15-QCD, SHA 621d2456...)
D10: 8.15 branch base = 9b507da (8d6da01 + R8.14 G4 test commit, additive). PDF-mandated base 8d6da01 is an ancestor.
D11: SU(3) Triton kernel lives in chromodynamic_grounding.py (additive module); qfhrr_kernels.py untouched (verified real Triton anchors @132/@156 — used for reference only).
D12: G3-QCD realized as fitted-gauge held-out SU(3) transport loss at D=65,536 (PDF cites phantom tests/test_henri_core.py).
D13: fixed deterministic DEFAULT_COLOR_PROJECTION [10,8] pre-registered constant.
Phantom CLIs: gpu_verification_suite.py --kernel su3_matrix_mul (file MISSING, #11), production_arc_run.py --mode phase815_benchmark (no --mode, #12), o_vsa_ingress_tokenizer.py --mode su3_binding_test (no argparse), efe_planner.py --mode verify_confinement (no argparse).
