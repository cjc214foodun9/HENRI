# PREREGISTRATION — Gate v2 fix, KSG MI metric, and the Dust ZO estimator

Author: HENRI arbiter. Date: 2026-10-05. Pin: 20261005.
Base SHA: 2163238622c32f7785a076d82c00cf966e8cc380 (pushed, remote == local).
Status: FROZEN BEFORE the paired measurement. Not to be edited after a run starts.

## Authority

2026-10-05 user message body approves: (1) a discriminating full-scale metric,
(2) the n_mem=256 investigation, (3) the versioned gate fix for C1/C2/C3,
(4) pushing the pending review commit. The Dust directive is a build request.

## Part 1 — Gate v2 (C1/C2/C3) in a NEW file

`henri_core/gates_v2.py`. `gates.py` is NOT modified: a committed receipt's
number belongs to the exact code state it names.

| Fix | Change | Boundary |
|---|---|---|
| C1 | non-finite R² → BLOCKED; no clamp | `min(1.0, nan) == 1.0` removed |
| C2 | verdict keyed on `estimator_sane`, never the raw value | a BLOCKED arm cannot pass |
| C3 | pass denominator read from the run, not a literal | `>= 4` removed |
| F4 | control needs a MARGIN (`margin=0.10`), not a 1-seed cutoff | |
| F5 | the shuffled-pairing control enters the verdict | it was computed, never used |
| F11 | explicit vocabulary PASS/FAIL/VACUOUS/BLOCKED | a passing control = VACUOUS |

Acceptance: `gate_u4_retention_v2` reproduces the v1 estimator values on the same
system (same `r2`), and flips to BLOCKED when handed a non-finite value.

## Part 2 — n_mem investigation (mechanism, already measured)

`resolve_n_mem` returns `n_mem = d_model // d_k`, so `n_mem · d_k == d_model`
in BOTH rules:

| d_model | old n_mem × d_k | Spec B n_mem × d_k |
|---|---|---|
| 128 | 32 × 4 = 128 | 4 × 32 = 128 |
| 512 | 128 × 4 = 512 | 16 × 32 = 512 |
| 1024 | 256 × 4 = 1024 | 32 × 32 = 1024 |

Prediction to test: a metric sensitive only to AGGREGATE rank cannot separate the
arms (this explains the K-B6 full-scale result). A metric sensitive to the
SLOT/WIDTH structure can. Measure `collapse` and `ksg_mi` on both arms.

## Part 3 — KSG MI estimator (pre-registered constants)

Estimator: Kraskov–Stögbauer–Grassberger, algorithm 1.
Pinned: `k=5`, Chebyshev distance, coordinates standardized, self excluded,
counts strictly-less-than the k-th radius, `N=512` texts, `n_comp=32` PCA
components (TRAIN half only).

Controls, both required before any gate use:
- Positive: `Y = rho·X + sqrt(1-rho²)·noise`, `rho=0.6`. Analytic MI
  `-0.5·log(1-rho²) = 0.443`. Estimate must be within 0.15.
- Negative: `Y` independent of `X`. Estimate must be ≤ 0.10.
If either fails → `estimator_sane = False` → the gate is BLOCKED. No MI claim.

## Part 4 — Dust zeroth-order estimator (G-DUST-1)

Module `henri_core/dust_zo.py`, clean-room from the specification's §2.1 math,
cross-checked against `github.com/qlabs-eng/dust` @
`2fdb01ba91ca38368a2b6b31a2697d77512ad175` (`dust.py` sha256
`154a03a7…`, MIT). No upstream code is copied.

Gate G-DUST-1: `cos(G_dust, G_autograd) >= 0.85` at the largest K, PLUS
`cos_shuffled <= 0.10` PLUS `cos_pos_ctl >= 0.99`.

**Rule: do NOT raise K until the gate passes. A K chosen to clear the bound is a
fitted gate.** Report the K-sweep as measured.

GPU gates, marked BLOCKED locally (no new GPU spend authorized, and `henri_core`
has no device plumbing):
- G-DUST-2 peak VRAM ≤ 2.20 GiB — **BLOCKED (needs a device path that does not exist)**
- G-DUST-5 ≤ 50 µs/step roofline — **BLOCKED (same)**

Not claimed anywhere: 38.4 µs/step, 87.5% traffic reduction, sub-microsecond
learning, BaTiO₃ operation. These are the document's aspirations (HYPOTHESIS).

## Part 5 — Governance

Defaults preserved. Additive only. `dk_target=32`, `n_mem`, and the sealed
447,145,231-param receipts are unchanged. Dust is default-OFF behind
`HENRI_DUST_ZO=1` / an explicit CLI flag. No backpropagation is deleted or
disabled. Push of the new work needs separate approval. No merge to main.

## Kill conditions (pre-registered)

- K1: `gate_u4_retention_v2` returns a non-BLOCKED verdict on a non-finite value → C1 fix failed.
- K2: `ksg_mi_selftest` positive control outside ±0.15 or negative > 0.10 → MI metric BLOCKED, discard.
- K3: `cos_real(Kmax) < 0.85` → G-DUST-1 FALSIFIED at that K; report the sweep, do not retune.
- K4: `|collapse_a - collapse_b| < 0.20` → the per-slot metric does not discriminate; G-U5 fails its own purpose.
