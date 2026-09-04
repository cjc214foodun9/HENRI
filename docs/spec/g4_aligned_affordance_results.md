# Carrier G4 — Functionally Aligned Sparse Affordance: Results

**Directive:** `Sprint_Closeout_Synthesis___Carrier_G4_Master_Directive.md`
(`HENRI-DIR-2026-09-V3-SPRINT-CLOSEOUT-G2-G3-G4`, SHA `bb25dfe247…`, 18,288 B) +
`Project_HENRI_V3_Carrier_G4_Master_Directive___Functional_Consistency_Synthesis.md`
(SHA `1203d7d8…`, 4,845 B).
**Prereg:** `docs/spec/g4_aligned_affordance_preregistration.md` (SHA `dcc09dcc…`, sealed `#fd47cb46` @1,174).
**Branch:** `feat/carrier-g4-aligned-affordance`; tested code @ `ceb06c4`.

## Verdict

**`G4_AFFORDANCE_FIT_COLLAPSE` — PG1 pre-flight kill, 0 live steps. 24th sealed falsification (25th carrier; 0 solved envs).**

Gate PG1 (min_action_auc ≥ 0.8800 on the N=128 action-stratified subset): **min_auc_subset 0.4** ✗ (binding; directive Failure Action = HALT_IMMEDIATELY_SEAL_FALSIFICATION). PG2 PASS (norm_drift 1.19e-7 ≤ 1e-6). PG3 PASS (k=64/8192 variance rule). G1–G4 live gates NOT RUN (PG1 pre-flight kill).

| action | n | moving | AUC subset | AUC full |
|---|---|---|---|---|
| 0 | 267 | 56 | **1.0000** ✓ | 1.0000 |
| 1 | 257 | 56 | **1.0000** ✓ | 1.0000 |
| 2 | 244 | 48 | **1.0000** ✓ | 0.9863 |
| 3 | 290 | 65 | **1.0000** ✓ | 0.9773 |
| 4 | 64 | 15 | **1.0000** ✓ | 1.0000 |
| 5 | 361 | 74 | **0.9333** ✓ | 0.9914 |
| 6 | 53 | 10 | **0.4000** ✗ | 0.2000 |

## Launch history (identical preregistered bounds: seed 20260927, 12 envs × 150, τ_stall 0.90, k=64, ridge 1e-2)

1. **`3e5bbe6` — harness defect** (`G4_HARNESS_DEFECT_ONESHOT_DEVICE`): `onehot` stayed on CPU while `y` was CUDA → `mask & (y == 1.0)` cross-device RuntimeError at main; 0 live steps. Fixed: `onehot → device`. Relaunched with identical bounds, no quarantine.
2. **`ceb06c4` — GENUINE kill** (sealed): receipt SHA `25cdfc5d…`, log SHA `30be942e…`.

## Mechanism finding (DERIVED)

- **C1 functional homology + sparse top-k REPAIRED every well-supported action.** G2's failures (a0–a3 at 0.68–0.74 AUC) now score **1.0/1.0/1.0/1.0** on the subset (0.99/0.98 full). The G4 root-cause resolution matrix is validated: fit/score functional identity + noise-floor suppression by top-k=64 works.
- **Action 6 is a small-sample block-selection failure, not a homology failure:** n=53, only **10 moving samples** → per-block displacement variance over 10 rows across 8,192 blocks is noise-dominated → the top-k mask selects noise blocks → residual inverted (subset AUC 0.4, full 0.2). The same mechanism that fixed a0–a5 cannot estimate 8×8 ridge transitions on 10 samples.
- **Labels verified** (match G2 calibration exactly: 0.19–0.234 moving/action); PG2 geometry clean. Kill genuine.

## Next-carrier levers (NOT authorized; require a new directive/prereg)

1. **Per-action minimum moving-sample gate** (e.g. ≥ 40 moving rows for top-k estimation; fall back to the D=64 bridge for under-supported actions).
2. **Cross-action shared block prior** — pool variance across actions with per-action refinement (borrowed strength for a6).
3. **Stabilized top-k for small n:** shrinkage on the variance estimate (posterior mean with a global prior) instead of raw empirical variance.
4. **W0 (G3 wave-packet planner wiring)** remains gated on a PASSING G4-equivalent PG1.

## Evidence

- Receipt: `C:/Users/chan/AppData/Local/Temp/g4_gates_receipt_r2.json` (SHA `25cdfc5d85eb…`)
- Log: `C:/Users/chan/AppData/Local/Temp/g4_launch_r2.log` (SHA `30be942e6981…`)
- Remote: `/tmp/henri_g4_aligned/g4_gates_receipt.json`, `g4_launch.log`
- Engine: `HENRI V2/experiments/verification/arc_g4_aligned_engine.py` @ `ceb06c4`; tests `test_g4_aligned_engine.py` (12/12 local + remote CUDA)
- Regression: 1,079 passed / 6 skipped (G4 +12, no regressions)
