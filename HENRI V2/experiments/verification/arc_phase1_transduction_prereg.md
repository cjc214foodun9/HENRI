# Phase 1 Pre-Registration — Minimal Viable Transduction

**Gate contract:** `SPEC-2026-10-01-PHASE1-TRANSDUCTION`
**Branch:** `feat/phase1-transduction` (from `main` @ `a039095`)
**Worktree:** `C:/Users/chan/henri-worktrees/phase1-transduction`
**Source specifications:**
- `project_henri_mvp_architecture_and_golden_standard_research_laboratory_specification.md` §5
- `project_henri_architectural_refinements_blackwell_roadmap_and_autonomous_research_lab_specification.md` §5

This document is written BEFORE implementation, per the specification's own
requirement (§4.2 item 1, "Pre-Registration of Evaluation Criteria").

---

## 1. Premise audit (REQUIRED before implementation)

The specifications state defects. Each claim is resolved against the live code.
Evidence class: OBSERVED (read from the working tree at `a039095`).

| # | Specification claim | Verdict | Evidence |
|---|---|---|---|
| 1 | `TapeLearner` requires `[B, T]` with vocab 257 | **PHANTOM** | No `TapeLearner` symbol in any tracked `.py` file. The named producer of "Gap 1" does not exist. |
| 2 | `SpatialCliffordTokenizer` must be created | **PARTIALLY TRUE** | No such module. But `o_vsa_torus_encoder.py` already emits the specified formula `Psi = sum v(x,y) exp(i(k_x x + k_y y))` with a MEASURED exact-roll operator (err 2.4e-05 at S=32). |
| 3 | 0 of 17 environments expose demonstration pairs | **PHANTOM (contested)** | `arc_demo_preflight.py:5` states "Never fabricates demos". The zero count is a committed governance result, not a loader defect. |
| 4 | Dense Koopman `[D, D]` causes the memory wall | **TRUE (already mitigated)** | No `[D, D]` allocation exists on the transition path. `LowRankCoupledTransition` (`efe_planner.py:70`) is already rank-factorized. |
| 5 | Linear ceiling 0.280 vs threshold 0.920 | **TRUE** | Present in `experiments/verification/arc_phase839_codec_repair_verdict.md`. |
| 6 | `tau = 0.038`, `beta = 26.3` | **TRUE** | `henri_probe_calibration.py:524` `FITTED_TEMPERATURE_60 = 0.038316`; `beta = 26.10`. |
| 7 | `Delta_Sagnac > 0.35` homodyne veto | **TRUE** | `arc_sagnac_veto.py:38` `DEFAULT_EPSILON_HARD = 0.35`. |
| 8 | `henri/ingress/` package layout | **FALSE** | Repository is flat: `HENRI V2/*.py`. No `henri/` package exists. |

**Consequence.** Action items 1 and 2 of the specification rest on phantom
premises. They are NOT executed as written. The bounded real residue is:
the rank-8 Clifford ingress gap (item 1) and the factorized-kernel contract
test (item 3).

---

## 2. The one measured gap

| Property | Specification | `TorusIngressEncoder` (live) |
|---|---|---|
| Blocks | 8,192 | 8,192 |
| Complex slots per block | **8** (rank-8 Clifford) | **4** |
| Total complex amplitudes | **65,536** | 32,768 |
| Position code | `exp(i(k_x x + k_y y))` | same (quantized `Z_S`) |
| Exact roll operator | required | measured 2.4e-05 |

The deficit is slot cardinality: 4 of 8. This is the single implementable
Phase 1 ingress delta.

---

## 3. Pre-registered gates

Seeds: `20260914` (inherited from the measured torus basis), `20261001`.
All gates run on CPU. No GPU latency claim is made in this phase.

| Gate | Statement | Kill criterion |
|---|---|---|
| P1-G1 | `SpatialCliffordTokenizer.encode` returns `complex64 [8192, 8]` | shape or dtype mismatch |
| P1-G2 | Encoded field is unit L2 norm over the flattened 65,536 amplitudes, err <= 1e-5 | norm drift > 1e-5 |
| P1-G3 | Cyclic canvas roll is an EXACT wave operator: `enc_raw(roll(X)) == M * enc_raw(X)`, err <= 1e-4 | err > 1e-4 (the property is lost) |
| P1-G4 | 1D byte input raises; only 2D integer grids are accepted | silent acceptance of a flat sequence |
| P1-G5 | Encoding is bit-deterministic across two instances | any nonzero entropy between runs |
| P1-G6 | `FactorizedTransitionKernel` allocates no `[dim, dim]` tensor | any tensor with `numel >= dim*dim` |
| P1-G7 | Per-action footprint is `2 * r * D * 8` bytes = 268,435,456 B at `D=65,536, r=256` | arithmetic mismatch |
| P1-G8 | `K = Q Q^H` with `Q` orthonormal is idempotent (`K^2 = K`), err <= 1e-4 | non-idempotent (algebra wrong) |
| P1-G9 | 512-step unroll holds bounded memory (no growth per step) | monotone growth |

Negative control (tautology guard): P1-G8 runs against a random `Q` arm that
must FAIL idempotence, and an orthonormal arm that must PASS. A suite where
both arms pass is rejected as vacuous.

---

## 4. Declared interpretation of the "300 MB" target

The specification states "parameter footprints under `300 MB`" and its own
arithmetic gives `2 * r * D * 8 B = 268.4 MB`. That figure is **per action**.
For `|A| = 8` the total is 2.15 GB, which the specification does not state.

**Decision:** the 300 MB bound is read as PER ACTION and is tested as
PER ACTION. The 8-action total is reported separately and is NOT claimed to
meet the 300 MB bound. The specification is self-inconsistent here; this
documents the resolution rather than silently choosing one.

---

## 5. Scope boundary

IN scope (executed): P1-G1..G9, module `HENRI V2/henri/ingress/spatial_tokenizer.py`,
module `HENRI V2/factorized_transition_kernel.py`, tests, harness isolation audit.

OUT of scope (requires separate authorization): Docker/NVIDIA Container Toolkit,
TimescaleDB + pgvector, Vast GPU dispatch, Triton kernels, any promotion to
`main`, any latency figure, any external benchmark score.

---

## 6. Round-trip extension (added 2026-10-01, AFTER the section-3 gates passed)

Gate contract addendum: `SPEC-2026-10-01-PHASE1-TRANSDUCTION/roundtrip`.
Module under test: `HENRI V2/tests/unit/test_phase1_roundtrip.py`.

Purpose: prove the discrete -> continuous -> discrete loop CLOSES on CPU at
reduced dimension before any GPU or ARC attempt.

    grid --encode--> psi [D complex] --TRANSITION--> psi' --SNAP--> index --decode--> grid'

### 6.1 Declared reduced-D configuration

| Parameter | Value |
|---|---|
| num_blocks | 256 |
| block_slots | 8 (the specification's rank-8 block) |
| modulus (canvas) | 8 (roll arm) / 16 (decode arm) |
| D complex | 2048 |
| D real (interleaved) | 4096 |
| beta | 26.10 = 1/tau with tau = 0.038316 |

### 6.2 API contract, established by executed probe (not by docstring)

`ContinuousHopfieldCleanup(dim=D)` is a REAL-space module of width D. For
COMPLEX waves of complex dimension Dc the contract is **dim == 2*Dc**:

- `store_engrams(complex [M, Dc])` -> float32 interleaved `[M, 2*Dc]`
  (`view_as_real`); the imaginary part IS preserved. Probed: `matches
  view_as_real(Im preserved)? True`.
- `retrieve(complex [Dc])` -> `(complex [Dc], weights [M])`; the dimension IS
  preserved. Probed: input 32 -> output `(32,)`.

Calls that pass `dim == Dc` for a complex-Dc wave are MIS-SIZED. That produces
a `(Dc/2,)` output and an apparent "dimension halving", which is a CALL-SITE
error, not a module defect. The gate pins the contract (RT-G2/R5) so the
mis-sized form cannot be reintroduced as a "discovery".

### 6.3 Gates (names match the test functions EXACTLY)

Doc–code alignment is enforced: each gate below names the real test in
`tests/unit/test_phase1_roundtrip.py`.

| Test | Statement | Must-fail control |
|---|---|---|
| `test_r1_codebook_index_recovery` | `snap(encode(g_j))` -> index j, 8 candidates | — |
| `test_r1b_foreign_wave_and_random_codebook_control` | a wave outside the codebook, and a RANDOM codebook, both retrieve less confidently | self 1.000 vs foreign 0.406 vs random 0.114 |
| `test_r2_noise_tolerance_sweep` | recovery vs eps in {0, .05, .10, .20, .40} | mild-noise floor >= 0.75 |
| `test_r2b_argmax_is_beta_invariant_weight_entropy_is_not` | argmax identical across beta; entropy not | **proves a top-1 result CANNOT validate tau** |
| `test_r3_roll_transition_arm` | `snap(apply_roll(psi_j))` == index of `roll_canvas(g_j)` | operator err <= 1e-4 |
| `test_r3b_wrong_sign_roll_misses` | the conjugate multiplier must be a different wave | distance > 1e-2, worse retrieval |
| `test_r4_norm_preserved_at_every_stage` | unit L2 at encode, transition, after snap | err <= 1e-5 |
| `test_r5_adapter_contract_exact_inverse` | complex -> real interleaved -> complex EXACT; agrees with module `_flatten` | dis-agreement fails |
| `test_r5b_layout_mismatch_degrades_retrieval` | a concat (re|im) layout must retrieve worse than interleave | wrong layout must degrade |
| `test_r6_determinism_via_run_manifest` | two manifests identical; distinct components distinct | uses `henri/determinism.py` (not dead code) |
| `test_r7_capacity_declaration_and_no_extrapolation` | M/D sparse at reduced D; production D BLOCKED | asserts D != 65536 |
| `test_r8a_index_decode_is_exact_by_construction` | decode via codebook INDEX is exact | — |
| `test_r8b_decode_canvas_cell_accuracy_measured` | lossy probe accuracy MEASURED, floor 0.70 | below floor fails |
| `test_r9_unseen_transformed_state_recovers_at_chance` | **boundary**: transformed state ABSENT from the codebook recovers at ~chance | rate < 0.5 |

### 6.4 Measured results (OBSERVED, CPU, this session)

- R1 codebook index recovery: 8/8.
- R1b controls: self 1.0000 vs foreign 0.4064 vs random-codebook 0.1136.
- R2 noise tolerance at beta 26.10: `{0.0: 1.0, 0.05: 1.0, 0.10: 1.0, 0.20: 1.0, 0.40: 1.0}`.
- R2b **beta-invariance**: argmax identical `[5, 5, 5, 5]` across beta in
  {26.10, 8.0, 2.0, 0.5} while weight entropy moved `0.0 -> 2.7624`. This is
  the evidence that a top-1 round-trip CANNOT validate tau.
- R3 roll-transition arm: snap(apply_roll(psi_j)) == codebook index of
  roll_canvas(g_j), operator error ~3e-07 (P1-G3 re-confirmed at this D).
- R7 capacity: M=16, D=2048, M/D=7.81e-03, crosstalk bound 0.0520.
- R8b decode_canvas mean cell accuracy 0.9355 (floor 0.70).
- **R9 boundary: 7/40 = 0.175 vs chance 0.125.** The loop does NOT generalize
  to a transformed state absent from the codebook.

**Decode accuracy is CANVAS-SIZE DEPENDENT — the floor is config-bound.**
Measured over 40 seeds per config (`pin_gate_floor.py`, this session):

| canvas | cells | vocab | min | mean | max | chance |
|---|---|---|---|---|---|---|
| 8x8 (GATE CONFIG) | 64 | 9 | **0.8594** | 0.9391 | 0.9844 | 0.1111 |
| 8x8 | 64 | 6 | 0.8125 | 0.9047 | 0.9844 | 0.1667 |
| 16x16 | 256 | 9 | **0.5000** | 0.5601 | 0.6016 | 0.1111 |

The `DECODE_FLOOR = 0.70` in the gate is pinned from the 8x8 minimum and is
NOT transferable: at 16x16 the same probe bottomed at 0.5000 and the floor
would fail. Holographic cross-cell crowding grows with the cell count, which
is exactly why the floor must be quoted WITH its canvas. Any future change to
`MODULUS` in the gate invalidates `DECODE_FLOOR` and requires re-measurement.

### 6.5 Honest limits of this extension

1. The identity arm is tautological BY DESIGN and is retained only as the
   declared control. The roll arm (R3) is the real loop proof; it reuses the
   P1-G3 exact-operator result rather than assuming it.
2. `decode_canvas` is holographic and LOSSY. Its accuracy is measured per run
   and pinned at a floor; it is never asserted to be exact.
3. No learned transition is tested. R3 uses a FIXED, analytically exact
   operator. A learned rank-r kernel is the next step and is NOT validated here.
4. CPU only at reduced D. No latency claim, no GPU claim, and **no
   extrapolation to D=65,536** (the M=10000/D=65536 gate remains UNMET).
5. R9 is a LIMITATION, not a win. Phase 1 closes the loop for states present in
   the codebook. In-context task compilation is what would extend it, and that
   is NOT established by this document.

## 7. Environment artifacts (same commit)

- `Dockerfile.phase1-cpu` — CPU verification image (python:3.11-slim + CPU
  torch). Built and RUN: **55 passed in 10.59s**, container exit 0.
- `.dockerignore` — excludes checkpoints, data, telemetry, vendor trees.
- `Dockerfile.vast` (pre-existing, NOT modified) — GPU image for Vast.

The CPU image proves dependency closure and reproducibility. It does NOT prove
model performance, GPU behaviour, or latency.

