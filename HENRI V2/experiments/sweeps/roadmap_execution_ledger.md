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

## 2026-08-16 — Roadmap R8.16 pre-registration (from Project HENRI V2 Strategic R&D Roadmap.pdf, SHA 0ca9f7a1...)
R8.16 = egress unbinder obstruction term L_obstruct + Triton LUT (~50 us target).
Status: PENDING OWN PDF / explicit go-ahead. Pre-registered acceptance: L_obstruct
must change candidate ranking on held-out ARC-style inputs (not rescale-all); Triton
LUT must beat the Phase 8.15 SU(3) kernel sustained interval 38.6 us @ D=65,536.
NOTE: Phase 8.15 G4-QCD (38.6 us sustained) already covers the Triton-latency
budget portion; the L_obstruct egress term remains the genuinely new work.

## 2026-08-16 — Phase 8.16 Egress Spec deviations (D14/D15) + phantom CLIs #13-#16
D14: spec cites `functor_flow.py` (phantom; real `henri_functor_flow.py`) and
     `gpu_verification_suite.py` (phantom #13), `--mode verify_lobstruct` (#14),
     `--mode test_unbinder_recall` (#15), `--mode phase816_benchmark` (#16 —
     production_arc_run.py has only --envs/--steps).
D15: G1-EGRESS threshold 1e-4 is BELOW the as-shipped metric's own noise floor
     (2*scale_eff*dim/latent = 2/3 with default Linear init, D-INDEPENDENT);
     probe @D=4096: L_valid 3.2e-2 >= 1e-4, ordering INVERTED (valid > mism).
     G1 FALSIFIED as specified -> phase verdict KILL (pre-registered criterion).
     Corrected-gate proposal (single shared projection + pinned scale band) in
     phase816_egress_design.md §3 — requires user approval, NOT auto-applied.

## 2026-08-16 — Phase 8.16.1 Shared-Projection Reform (PDF bdf4602b...)

D14 (8.16): G1 noise floor 2/3 vs 1e-4 = unfalsifiable gate; KILL sealed e0745d7.
D15 (8.16): phantom CLIs #13-#16 confirmed missing.
D16 (8.16.1): SPEC CONSTANT ERRORS — claimed L_valid 2.7e-5 / L_mism 3.2e-4 @ s=5e-6
   require ||w-a||^2 ~ 1e7 (impossible); actual (probe OBSERVED) L_valid = 0.0 exact,
   L_mism = s^2*k (D-independent, 5.1e-8 @k=2048); "2s^2" omits k/D projector factor.
   GATES UNCHANGED and satisfiable: G1 (0 < 1e-4), G2 (ratio inf >= 10).
D17 (8.16.1): phantom CLIs #17 (functor_flow.py — real: henri_functor_flow.py) and
   #18 (gpu_verification_suite.py --kernel phase_ring_lut_unbinder) confirmed missing;
   spec test path tests/test_phase816_1_calibration.py -> repo convention tests/contract/.

## 2026-08-16 — Phase 8.17 In-Context Task Alignment (PDF 1342944c...)

D18: spec C1 reference code has an indentation bug (7-space body) — fixed at implementation (spec-exact semantics).
D19: G1 gate is dtype-blind — c64 SVD rounding floor over 8192 blocks ~5e-5 exceeds literal 1e-6 even for exact algebra; pre-registered dual-dtype (c128 = math gate <1e-6, c64 = live fidelity <1e-3). G2 oracle must be det-1 SU(3) (live matrix-exp producer); random-QR oracle FAILS (112.2, phase artifact). G3 141x >= 100x. G4 standing BLOCKED_NO_DEMONSTRATIONS; C2 goal bridge typed BLOCKED_MISSING_FIELD_WAVE_TRANSDUCER (no field->wave transducer exists). Phantom CLIs #19-#21 (--mode verify_procrustes_compiler, --mode test_anisotropic_creep, --mode phase817_live_benchmark).

 ## 2026-08-17 — Phase 8.18 SU(3) Field→Wave Transducer (spec 158c02c7..., brief 19f27caa...)

 D19: `torch.linalg.matrix_log` absent in PyTorch 2.12.0+cu130 (local AND remote) → eigendecomposition fallback (exact for unitary).
 D20: projection einsum index collision — spec `abc,bncb->bna` corrected to `aij,bnji->bna` (field-trace over generator pairs).
 D21: Cardano per-branch pairing `v_k = -p/(3u_k)` mandatory — unpaired roots error 0.79 → 1.25e-6 (probe OBSERVED).
 D22: latency gate unified to 50.0 µs (gate table wins over §2.3's 45 µs).
 D23: CUDA-only dtype promotion — Python `1j` × c64 → c128 on CUDA (weak-scalar promotion differs CPU/CUDA) → einsum type error; explicit `.to(basis.dtype)` at every complex construction site (C1 `_matrix_log`, runner `rand_su3`, commuting control). CPU probes CANNOT catch this class (2nd CUDA-only trap of the phase).
 D24: Triton JIT rejects nested comprehensions → `_su3_log_kernel` fully unrolled via Cayley–Hamilton projectors `P_k=(U²−S_k U+P_k I)·s_k`; probe-4 validated 3.659e-07 / 3.008e-06 vs eig-log BEFORE rewrite.
 D25: remote Triton is 3.7.0 (brief claims 3.1 — probe-beats-document); `tl.math.atan2` AND `tl.atan2` absent → libdevice `_tlib.atan2` (version-safe import).
 D26: float32 principal-root cancellation — sqrt(D) radicand rounds −1 ulp below |D_r| → NaN poisoned 208/8192 rows (290 NaN elements); clamp radicands ≥ 0 (mathematically always ≥ 0).
 D27: Triton kernel [N,18] row-major buffer addressed with stride 1 instead of stride 18 — rows 1+ read/wrote wrong flat offsets (row 0 matched by coincidence 0*18==0); remote bisect + raw-load probe isolated the mim11 load divergence; stride-18 fix → kernel-vs-eig 1.434e-06.
 Phantom CLIs #25-#26 (8.19 brief): `--mode verify_mcts_planner`, `--mode phase819_live_gauntlet` — production_arc_run.py exposes only --envs/--steps; real self-test via `su3_mcts_planner.py __main__`, live via the real runner interface (8.19 carries the fix).

 EVIDENCE (OBSERVED @ 69313c8, PID 78811, remote RTX 5090, JSON sha256 b51a0e0902445ba8981b184d01de39a8a1263787bdea1f15f5937fd56637ec5a):
 G1 c128 5.485e-07 / c64 8.093e-07 (PASS < 1e-5); G2 227.38 vs commuting control 1.34e-03 (PASS > 0.5); G3 latency 27.79 µs (PASS ≤ 50) + log err vs eig 1.434e-06 (PASS ≤ 1e-3); G4 BLOCKED_NO_DEMONSTRATIONS (lp85/dc22/cn04 examples None — by design, 8.17 precedent).

 VERDICT: SEALED ACCEPT (CASE A per brief 19f27caa...; G1–G3 PASS, G4 blocked acceptable). Commit `69313c8` is the component commit; seal commit follows.

## 2026-08-17 — Phase 8.20 Action-Conditioned EFE Grounding (postmortem ddf90d16..., spec p820_brief 2fc28a54...)

C1 `henri_external_outcome_refactor_module.py` ActionOutcomeGeneratorStore (D_a in su(3)^8192, EMA lr=0.1, eig-log D19 fallback); C2 action-conditioned pragmatic EFE in efe_planner.py + darwinian_phase_swarm.py (goal-distance term, flag HENRI_ARC_ACTION_EFE default-OFF); C3 StationarityDissipationThermostat (T_a x1.5, N_repeat <= 2); C4 production_arc_run.py --mode phase820_live_gauntlet (phantom #29 -> real). D28 (16-color projection) applied at base.
D29: spec einsum 'na,abc->nabc' -> 'na,aij->nij' (D20/D21 class). D30: projection 'abc,nbnc->na' -> 'aij,nji->na'. D31: matrix_log absent -> eig fallback (standing D19). D32: gell_mann_basis.to(theta.dtype) cast complex basis to float32, discarding imaginary generators lambda2/5/7 -> fit error 1.776; keep basis complex, cast theta up. RULE: never .to(real_dtype) a complex Gell-Mann basis.
D33: Scenario C fix — p820_update_info initialized at outer step scope (nested post-observation C1 block skipped before first emit -> UnboundLocalError at line 1794, OBSERVED remote launch log; init None fail-closed with p820_var_efe). Commit 8334cb4.
D34: postmortem Step-1 CLI `--mode verify_cuda_generators` is a PHANTOM (module implements `verify_action_generators`); ran the real mode. Postmortem Step-1 CLI `--mode phase820_live_gauntlet --envs 20` is real (implemented C4).

EVIDENCE (OBSERVED @ 78f028b + 8334cb4, remote RTX 5090):
G1-8.20 Var_a(G) >= 0.0100: local CPU 0.00953 -> PASS at EMA convergence (contract test); remote CUDA contract suite 7/7 PASS (11.20 s).
G2-8.20 fit err < 0.0500: remote verify_action_generators fit error 0.000001 (gate < 0.0500), identity 0.000000, Lie separation 38.60 -> PASS.
G3-8.20 N_repeat <= 2: contract sequence 00111111 N_repeat=2 -> PASS (remote 7/7).
G4-8.20 live progression > 0: MEASURED 0 (see telemetry below) -> FAIL.

G4 TELEMETRY (OBSERVED, PID 81020, log sha256 5a950b35..., JSONL sha256 c1a1eaad...): 93+ steps, 100% GameAction.ACTION6 (74/74 step records), 0 levels completed, 0 WIN, single env ft09-0d8bbf25 never left (20 envs requested), admissible_count=1 on every record, phase820_var_efe=None and phase820_update_info=None on all 74 records (len(efes)<2 -> variance uncomputable; C1 update block never engaged), 0 phase820 errors in log. ACTION6 + EFE +2.000 stationarity stall reproduced EXACTLY as in 8.19 SEALED KILL 5d73afb.
Root cause: the live legal-action mask admits only ACTION6 (admissible_count=1) -> the action-conditioned generator never has >1 candidate to differentiate; the flat-EFE/stationarity failure mode is upstream of the C2 machinery (env-legal action space, not the EFE landscape).

VERDICT: SEALED KILL (postmortem Scenario B: G1-G3 PASS & G4 = 0). Retain D28 (16-color projection) + D32 (complex basis preservation) as shared production fixes. Commits 78f028b + 8334cb4 are the component commits; seal commit follows. Phase 8.21 direction per postmortem: search horizon depth + legal-action-space diagnosis.

## 2026-08-17 — Phase 8.23 IN-FLIGHT (spec 75e66d14…, base 6ba8f76, branch feat/phase823-target-grounding-action-alignment)

Implemented + pushed `0388486`: C1 `synthesize_demonstration_goal_wave` (block-SVD Procrustes W_task → field transport → goal wave) + G1/G2 verify (pragmatic gradient 0.5219 ≥ 0.0100 PASS; fit loss −87.0% > 50% PASS); C2 `verify_opine_mcts` (G3 engagement 0.90 ≥ 0.25, unitarity 5.9e-07 PASS) + live SagnacMCTSPlanner instantiation with dual-channel-veto causal consumer (D39: full search() blocked by design — no held-out target); C3 `HENRI_ARC_TARGET_GROUNDING` default-OFF + `phase823_live_gauntlet` + goal consumer activation (fixed real bug: per-env lambda_goal assignment zeroed the constructor activation). Local 23/23 contracts PASS. CUDA verify + G4 gauntlet QUEUED behind 8.22 gauntlet run2 (PID 88153 @ 6ba8f76, GPU exclusivity). SOTA blockers resolved by this phase: #2 (live planner), #3 (goal grounding), #4 (action generator learning), #1 (OPINE live telemetry); residual: #5 CEGIS stub, #6 Zone C recall, #7 transition model, #8 action-head calibration.
