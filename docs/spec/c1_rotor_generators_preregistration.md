# Carrier C1 Pre-Registration — Factorized SO(8) Rotor Action Generators

**Carrier:** `C1_FACTORIZED_SO8_ROTORS`
**Directive:** `Carrier_C1_Master_Directive_SO8_Rotor_Action_Generators.md` (inbox SHA-256 `2554c3fc4f2169bc5324219f91b839653134ce99a35972518e7fcc70ee728814`, 12,417 B, read in full)
**Sealed base:** `b0f76ab` (`feat/carrier-g8-subgoal-steering`, G8 Phase B results — `G8_FALSIFY_SUBGOAL_REACHABILITY`, seal `#23a51381`, ledger @1,223)
**Branch (this carrier):** `feat/carrier-c1-rotor-generators` (worktree `C:/Users/chan/henri-worktrees/carrier-c1`)
**Status:** SEALING — prereg written before engine code (TDD order).

## 1. Objective

Falsify or verify, on the remote RTX 5090 CUDA target only, that replacing
passive scalar action biases with exact per-action SO(8) rotors
(`Ψ' = R_a Ψ`, Cayley-generated, block-wise on the live `[num_blocks, 8]`
Cl(3,0) wave) produces measurable action→state displacement that survives
into the live arcade loop — against the sealed P2-0 baseline
(`P2_NO_PROGRESS` #81cf42c1: mean Δν 2.07e-4, 395 creeps, 0 advances, 0 solves).

## 2. Pre-seal corrections to the directive (disclosed, not silent)

1. **Shape boundary.** The directive's kernel sketch reshapes to `[B, 65536]`
   and calls `F.normalize(p=2, dim=-1)`. Live planner waves are
   `[num_blocks, 8]` real per-block unit rows (`||ψ_m||_2 = 1.0 ± 1e-6`);
   global renormalization would break the per-block invariant and silently
   introduce a new wave family. An orthogonal rotor preserves row norms
   exactly, so the engine rotates `[num_blocks, 8]` in place and performs NO
   normalization.
2. **Gate metric repair (C1_GATE_DISPLACEMENT_VARIANCE).** As literally
   written, the gate is `Var_{a}(||Ψ(a) − Ψ||_2) ≥ 0.05` on the raw tensor
   norm. The frozen-seed probe (CPU fp32, seed 20260930, 8,192 unit rows)
   measured the literal value at σ = 0.15 at 11.71 — a dimension artifact:
   the raw norm scales with √N (per-row RMS ≈ 0.31 ⇒ raw norm ≈ 28), so
   the threshold has no fixed meaning across scales and the metric reads
   magnitude spread only. A magnitude-only variance over 7 samples cannot
   detect direction-distinct actions with equal displacement magnitude
   (false-kill class: two rotors of equal angle in different planes read
   Var ≈ 0 and would kill a healthy mechanism). Operationalized per the
   directive's stated intent ("actions must generate statistically distinct
   spatial translations"): **min pairwise per-row-RMS separation**
   `min_{a≠b} (1/√N) ||R_a ψ − R_b ψ||_F ≥ 0.05`
   on a calibrated random unit-row fixture, plus min per-action displacement
   `min_a (1/√N) ||R_a ψ − ψ||_F ≥ 0.02`.
   Both are measured on per-row-RMS (dimension-normalized) scale, matching
   the P2-0 Δν comparison scale. Measured at the frozen init: pairwise
   separation 0.387 ≥ 0.05, min displacement 0.308 ≥ 0.02 (PASS margin
   ~6–15×).

## 3. Mechanism

- Ambient space D = 65,536 partitioned into M = 8,192 Cl(3,0) blocks of d = 8.
- Per action a ∈ {0..|A|−1}: skew generator A_a ∈ 𝔰𝔬(8) from 28 upper-tri
  bivector floats; rotor R_a = Cayley(A_a) = (I − A_a/2)^{-1}(I + A_a/2),
  exactly orthogonal in exact arithmetic.
- Shared-rotor basis: one R_a per action applied identically block-wise
  (`ψ_m' = R_a ψ_m`) — the directive's stated 196-float configuration
  (7 actions × 28). NOT block-specific rotors (that would be 8,192×196).
- Init: `torch.randn(num_actions, 28) * 0.15`, seed 20260930 (frozen);
  zero-trainable at launch (nn.Parameter container, no optimizer path in
  this carrier).
- Injection seam (pre-seal correction 3, after live-surface audit): the
  P2-0/G8 coupling measurements ran through the G-series gauntlet
  (`arc_g7_calibrated_engine.py` launcher → `G4AlignedEngine.run_gauntlet`
  with the M1 Δν meter), NOT `production_arc_run.py`/EFEPlanner. For
  P2-0-comparable Δν the C1 policy therefore lives in the G-series scoring
  path: `C1RotorSteeringEngine(P1GoalSteeringEngine)` overrides
  `score_all_actions` to generate candidates by exact rotor rotation
  `cand_a = R_a ψ_full` (all 8,192 blocks) and scores with the P1
  potential-drop form `j(a) = (clamp(ΔV(a),−1,1)+1)·π_a^H`. Launcher routing
  in `arc_g7_calibrated_engine.main` under `HENRI_C1_SO8_ROTORS=1` (lazy
  import; flag absent ⇒ module never imported — default-OFF differential).
  The EFEPlanner `_c1_rotor_engine` hook is DEFERRED: no production-path
  change in this carrier.
- Environment action count: |A| = 7 per directive; runner reads live
  `decoder.id_to_action` length when available (min 7).

## 4. Bounded experiment scope (frozen)

| Parameter | Value |
|---|---|
| Environment cohort | Same 12 as P2-0/G8: ar25 sc25 tr87 cd82 lp85 wa30 ft09 g50t sk48 bp35 ka59 sb26 |
| Seed | 20260930 (same as P2-0 — policy determinism for direct Δν comparison) |
| Steps | 1,800 (12 envs × 150) |
| Flag | `HENRI_C1_SO8_ROTORS=1` (default-OFF) |
| Device | remote CUDA only; local CPU = contract/preflight, never the gate |
| Output | `/tmp/henri_c1_rotors/`, receipt `c1_gates_receipt.json` |

## 5. Pre-registered gates

| Gate | Criterion (per-row-RMS scale) | Verdict class |
|---|---|---|
| C1_GATE_ORTHOGONALITY | max_a ‖R_aᵀR_a − I₈‖_F ≤ 1e-6 | kill: `KILL_RUNNER_MATH_ERROR` (BLOCKED_INFRA) |
| C1_GATE_DISPLACEMENT | min pairwise sep ≥ 0.05 AND min per-action displacement ≥ 0.02 (fixture, corrected metric) | kill: `KILL_RUNNER_AFFORDANCE_COLLAPSE` (BLOCKED_INFRA) |
| C1_GATE_ACTION_COUPLING | mean Δν ≥ 0.02 over 1,800 live steps (P2-0 baseline 2.07e-4; 100× margin) | `C1_FALSIFIED_ACTION_COUPLING` if < 0.02 |
| C1_GATE_LG2_SOLVED | levels_completed ≥ 1 across cohort | `C1_FALSIFIED_TASK_SOLVE_LG2` if 0 |
| C1_GATE_LG3_LATENCY | mean step latency ≤ 2 ms (RTX 5090) | `FLAG_KERNEL_PERF_REGRESSION` (flag only, not seal basis) |

Verdict taxonomy: `C1_ROTOR_VERIFIED` (coupling ≥ 0.02 AND ≥ 1 solved) |
`C1_FALSIFIED_ACTION_COUPLING` | `C1_FALSIFIED_TASK_SOLVE_LG2` |
`FALSIFIED_NO_ENGAGEMENT` | `BLOCKED_INFRA`.

W0 remains GATED on ≥1/12 solved. A passing C1 does NOT open W0 unless
LG2_SOLVED passes. 0 solves with healthy coupling is a grounding result
(action semantics repaired; task/goal semantics next carrier), not a license
to add coherence terms.

## 6. Kill experiments

- K1 orth/norm: orth error ≤ 1e-6 AND per-row norm drift ≤ 1e-5 on an
  8-step unroll. Failure ⇒ kernel defect ⇒ BLOCKED_INFRA.
- K2 identity differential: flag absent ⇒ score_actions byte-identical
  (engine attribute never consulted). Contract test asserts
  `predicted == transition(state, wave)` when engine is None.
- K3 engagement: ≥ 95% of live telemetry rows carry `pred_source = c1_rotor`
  and finite displacement telemetry. Failure ⇒ `FALSIFIED_NO_ENGAGEMENT`.
- K4 shuffle control (remote): per-action rotor assignment permuted across
  actions; coupling Δν_shuffled − Δν_fit separation required ≥ 0.05 if the
  fit gate passes. Failure to separate ⇒ fit not causal ⇒ FALSIFIED.

## 7. Telemetry contract (per step)

`pred_source` (`baseline`|`c1_rotor`), `c1_displacement` (per-row RMS
‖ψ_t+1 − ψ_t‖), `c1_orth_err`, `c1_action_id`, `c1_engaged`, plus existing
`grid_dist`, `delta_nu` (M1 meter), `levels_completed`, sagnac fields.

## 8. What this prereg does NOT authorize

- No change to `main` or the default production path (flag default-OFF).
- No training / SGLD creep / optimizer path on the rotors in this carrier.
- No reuse of consumed P2/G8 cell telemetry as fresh evidence.
- No AAII/benchmark capability claims from a coupling pass.
- Remote gauntlet dispatch is a separate approval-gated step after local
  preflight + G-series regression pass at this SHA.
