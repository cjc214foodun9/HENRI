# Carrier C1 — Factorized SO(8) Rotor Action Generators: Live Results

**Verdict: `C1_FALSIFIED_ACTION_COUPLING` + `C1_FALSIFIED_TASK_SOLVE_LG2`**
(sealed 2026-09-03, live remote run; 30th sealed task-level falsification chain)

**Prereg:** `docs/spec/c1_rotor_generators_preregistration.md` (sealed before code)
**Authorization:** `Carrier_C1_Governance_Resolution_and_Execution_Authorization.md`
(SHA-256 `cc4852475d8a3a46e5cd77ddd5491082b090585c65ac1ab821277d0fdf6d5949`; threshold
rectification: `LIVE_GATE_C1_1_COUPLING` = 0.0200 binding; 0.0100 voided as doc typo)
**Directive anchor:** `2554c3fc…`; **Audit doc anchor:** `9e052fdb…`
**Commit under test:** `ed7b603c5784336e8ea48b83fafe5e3192b8d495` (origin
`feat/carrier-c1-rotor-generators`; remote detached worktree `/workspace/henri-c1-dispatch`, clean)
**Run:** 12 envs × 150 steps = 1,800, seed 20260930, `HENRI_C1_SO8_ROTORS=1`,
arc_g7_calibrated_engine.py → G4AlignedEngine.run_gauntlet, M1 Δν meter
**Receipt:** `/tmp/henri_c1_rotors/c1_gates_receipt.json`, SHA-256
`9ecfb9b3d24f7cbeb2e18401f9bf6121fcc5c2f7e687168b825164fd319870a5`
**Log:** `/tmp/henri_c1_rotors/c1_launch.log`, SHA-256
`d6424f180f4e6e62118ec142348eb90ecf5d11d49b5518a89a7c31ac220491b8` (EXIT:0)

## Abort attempt (preserved, not discarded)

Attempt 1 aborted at 300/1,800 (`G4_ARCADE_MAKE_NONE`, receipt
`c1_gates_receipt_abort_300.json`, log `c1_launch_abort_300.log`): the arc_agi loader
scans `environment_files/` relative to process CWD; the first launch lacked the root
cache and a live API download of tr87 hit `ReadTimeout` (three.arcprize.org). Infra
failure, NOT a scientific verdict. Fix: full 25-env cache at worktree root (loader
fell back to cache on later API refusals — run survived). Retry kept the ORIGINAL
sealed bounds (12×150, seed 20260930).

## Gate results (final run)

| Gate | Rule (prereg / auth doc) | Result | Evidence |
|---|---|---|---|
| Orthogonality (pre) | max_a ‖R_aᵀR_a − I₈‖_F ≤ 1e-6 | **PASS** | `c1_orth_err = 6.74e-07` |
| Engagement (K3) | ≥95% rows pred_source=c1_rotor | **PASS** | `policy_mode=C1_SO8_ROTOR_STEERING`, `c1_score_calls=1800` (=100%), `affordance_updates=1783` |
| LG1 coupling | mean Δν_wp ≥ 0.0200 | **FAIL** | `mean_delta_nu_wp = 2.238e-4` (P2-0 floor 2.07e-4; ~100× below gate) |
| LG2 solved | ≥1/12 envs | **FAIL** | `envs_solved = 0`, all 12 env_levels 0 |
| LG3 latency (flag only) | kernel ≤ 2.0 ms | **FLAG** | `c1_kernel_latency_ms = 5.42` (G8 pattern: 10.48) |

Calibration context: `pg1_min_auc=1.0`, pg2/pg3 true (bank calibration healthy);
`creeps=400`, `resets=17`, `waypoint_advances=0`; `wall_s=1152.0`.

## Verdict-ordering note (harness, not seal basis)

`_decide_verdict` (arc_c1_steering_engine.py:144-158) checks LG3 latency before LG1
coupling, so the receipt's first-fired symbol is `C1_GATE_LG3_LATENCY_FAILED`. Per the
sealed prereg, LG3 is flag-only (`FLAG_KERNEL_PERF_REGRESSION`, never seal basis);
both seal-basis gates (coupling, solve) failed regardless of ordering. The sealed
verdicts are `C1_FALSIFIED_ACTION_COUPLING` and `C1_FALSIFIED_TASK_SOLVE_LG2`. Future
carriers: order LG1 before LG3 in `_decide_verdict`.

## Interpretation

- Rotor kernel is mathematically correct (orth error 6.7e-7; fixture separation 0.387).
- Live selected displacement `c1_mean_selected_displacement = 0.0046` (per-row RMS,
  vs 0.308 fixture): rotors displace live arcade waves only weakly at the selected
  actions; goal-relative Δν sits at the P2-0 terminal-goal noise floor (2.24e-4 vs
  2.07e-4). Action semantics (SO(8) rotor generators) do NOT close the
  action→outcome coupling gap. Pattern reproduces P2-0/G8 at the action level.
- 0/12 solved. W0 stays **GATED** on a non-zero live task completion.
- 0 solves with healthy kernel geometry = grounding result (goal semantics next
  carrier), NOT a license to add coherence terms.

## Next actions

1. Do NOT reopen C1 tuning without new evidence (sealed-record discipline).
2. Record the falsification event with receipt SHA `9ecfb9b3…` as evidence.
3. Next carrier must address goal-relative displacement (not wave-space wiggle):
   candidate directions per master-plan ordering, each with own prereg + kill gates.
4. LG3 kernel-batching (448 tiny launches; 5.42 ms) = perf carrier, not seal basis.
