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
