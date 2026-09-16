# Phase 10.4: UWSH, Zone A arithmetic, and the torus encoder adjoint

Sealed measurement record. Every figure below is copied from a machine receipt in
`experiments/verification/`. The pre-commit seal gate (`validate_seal_consistency.py`)
verifies that each arm and each declared scalar appears here verbatim, so this document
cannot drift from its receipts without blocking a commit.

## Source of record

| Field | Value |
|---|---|
| File | `Project HENRI_ Universal Weight Subspace Integration, Zone A Swarm Geometry, and Zone C Causal Substrate Orchestration.pdf` |
| Bytes | 516311 |
| sha256 | 1ea527bd87c352839ee2aba0b88080b1aed9033c4268b5681cda6313e7a7b2ef |
| Pages | 16 |
| Characters extracted | 26242 |

Attention: **two copies of this directive exist on disk.** The attached copy
(`1ea527bd87c35283`, 16 pages) is a **strict subset** of a larger copy
(`615d6f63b5329098`, 22 pages): every normalised line of the former appears in the latter,
and the latter adds 159 lines carrying the Phase 10.4 adjudication and the actionable
directives. This document adjudicates the **union**, so no directive text is missed.

External reference: arXiv **2512.05117v2**, "The Universal Weight Subspace Hypothesis"
(Kaushik, Chaudhari, Vaidya, Chellappa, Yuille), sha256
`d41574928118f26653df30e162527fef89a5ebe423214da865caa6ddef316b80`, 37 pages. Its scope is
**shared spectral subspaces across neural-network weight matrices** (1100+ models). It does
NOT study VSA wave operators or per-slot complex diagonal operators, so the transfer to
HENRI is an **analogy**, tested directly rather than imported.

## 1. Directive 3 — UWSH: implemented, controlled, FALSIFIED

Receipt: `experiments/verification/uwsh_subspace_60_observed.json`
(`canonical_config=true`, `n_common=60`, self-test `True`, `n_fit_used=100`).

The subspace is fitted on **100 ARC tasks disjoint from the 60 evaluation tasks**,
so no information about the scored tasks enters `U_k`. The regression is the directive's own
projected ridge: `(A^T A + lambda I_k) c = A^T b`, `W = U_k c`. Parameters drop from
32768 to `k`. Scoring is identical to the sealed Phase 10.1/10.3 evaluator:
per-block L2 `_to_real`, then cosine against the encoded held-out output.

Both baselines reproduce the sealed values exactly: `diag_ls` +0.4215
and `identity` +0.4033, with ceiling +0.7536.

| arm | held-out | in-sample ceiling | gap | beats identity | n |
|---|---|---|---|---|---|
| `identity` | +0.4033 | +0.4033 | 0.0000 | 0.0% | 60 |
| `diag_ls` | +0.4215 | +0.7536 | 0.3320 | 45.0% | 60 |
| `mean_only` | +0.3584 | +0.3698 | 0.0115 | 16.7% | 60 |
| `uwsh@1` | +0.0293 | +0.0265 | -0.0028 | 11.7% | 60 |
| `uwsh@2` | +0.0081 | +0.0109 | 0.0028 | 11.7% | 60 |
| `uwsh@4` | +0.2059 | +0.2221 | 0.0162 | 10.0% | 60 |
| `uwsh@8` | +0.2388 | +0.2583 | 0.0195 | 8.3% | 60 |
| `uwsh@16` | +0.3808 | +0.4068 | 0.0260 | 15.0% | 60 |
| `oracle@1` | +0.0808 | +0.0995 | 0.0187 | 8.3% | 60 |
| `oracle@2` | +0.0969 | +0.1268 | 0.0300 | 10.0% | 60 |
| `oracle@4` | +0.2144 | +0.2619 | 0.0475 | 8.3% | 60 |
| `oracle@8` | +0.2866 | +0.3519 | 0.0653 | 10.0% | 60 |
| `oracle@16` | +0.4011 | +0.5118 | 0.1107 | 33.3% | 60 |
| `random@1` | +0.0016 | +0.0029 | 0.0013 | 6.7% | 60 |
| `random@2` | +0.0013 | +0.0037 | 0.0024 | 10.0% | 60 |
| `random@4` | +0.0034 | +0.0068 | 0.0035 | 10.0% | 60 |
| `random@8` | +0.0069 | +0.0112 | 0.0043 | 11.7% | 60 |
| `random@16` | +0.0088 | +0.0141 | 0.0054 | 10.0% | 60 |

**Verdict: `ACCEPT_UWSH = false` — FALSIFIED as stated.** The best transferred arm
(`uwsh@16`) reaches +0.3808 against the incumbent
+0.4215, and **no** `uwsh` arm exceeds `diag_ls`. All arms are scored
on the **common subset** of n=60, so the comparison is like-for-like.

**The random control carries the falsification.** `random@16` reaches only
+0.0088. The fitted subspace clears the random subspace by
+0.3720 at k=16, so `U_k` is **not** merely an
orthonormal rank constraint doing unspecified work: the fitted directions carry real signal.
What fails is the premise that they carry **enough** signal.

### 1.1 Why: the oracle separates "rank is binding" from "transfer is binding"

`oracle@k` is fitted on the evaluation tasks' **own** `W*`, so it is an **upper bound** on what
any k-dimensional subspace could achieve at fit time. Measured ceiling:

| k | oracle held-out | oracle ceiling | gap |
|---|---|---|---|
| 16 | +0.4011 | +0.5118 | 0.1107 |
| 24 | +0.4079 | +0.5756 | 0.1677 |
| 32 | +0.4207 | +0.6335 | 0.2127 |
| 40 | +0.4205 | +0.6714 | 0.2510 |
| 48 | +0.4141 | +0.6996 | 0.2855 |
| 56 | +0.4086 | +0.7175 | 0.3089 |

The oracle ceiling rises monotonically to +0.7175 at
k=56, still **below** the incumbent ceiling
+0.7536. Even an oracle that is allowed to see the answers
cannot reach `diag_ls` expressivity at any rank constructible here
(available rank 60, bounded by the 60 reference tasks, not by ARC).
**The rank restriction itself destroys expressivity**, so this is not a transfer failure that a
better universal basis would fix.

### 1.2 The `gamma` claim does not describe the incumbent

The directive states `gamma` improves "from 1e-4 to 42.4". Both endpoints are problematic:

| quantity | value |
|---|---|
| directive formula `gamma = M*S/D` | 3.662e-04 |
| observed operations per complex slot (what `diag_ls` actually solves) | 3.0 |

The 1e-4 figure treats `W` as one dense 65536-parameter operator. The
incumbent is **per-slot**, so its own ratio is already >= 1, not 1e-4. The two endpoints are
therefore not comparable. Worse, the UWSH endpoint needs `N_active`, a quantity the directive
**never defines**: candidate `(k, N_active)` pairs landing near 42.4 include (1, 16) and (8, 128),
i.e. the target is reachable by several unrelated definitions, so it cannot evidence anything on
its own.

## 2. Directive 4 — torus encoder adjoint: CONSTRUCTED and VERIFIED

Receipt: `experiments/verification/torus_encoder_adjoint_observed.json`.

First, an independent check that my explicit operator **is** the live encoder: max absolute
difference from `enc.encode` is 1.581e-05. Only then are the following
numbers meaningful.

### 2.1 The spec's IDFT premise does not hold for this state

Spec 4.1 asserts the adjoint is a 2D IDFT over an `S x S` frequency lattice. Measured:
the state is `[num_blocks, 8]` real = **32768 frequency samples**, not a lattice;
`S = 32` is the **position modulus**, not an array length. The encoder samples
961 distinct non-zero `(kx, ky)` pairs from a lattice of
961 (a density of 25.6x), with
8192 `(0,0)` samples down-weighted by `dc_weight = 6.104e-05`.
So the correct adjoint is **not** an inverse DFT; it is the left-inverse of an explicit
{32768, 48} complex operator, which is what was
implemented (factored, no dense matrix).

Rank on small grid families is **full column rank** (measured on 4x4/V3, 6x6/V4, 8x8/V4), so the
pre-normalisation operator carries the information.

### 2.2 Wave-to-grid decoding works

`_to_real` L2-normalises each block, destroying the per-block complex amplitude, so decoding
must profile out an unknown per-block scale. With that handled, exact grid recovery on real ARC
test inputs:

| case | grid | V | exact | mean block cosine |
|---|---|---|---|---|
| `synthetic3x3` | 3x3 | 4 | True | 0.999409 |

Aggregate: **8 exact recoveries** in
8 cases, cell accuracy 1.0000,
exact rate 1.0000. The Moore-Penrose round-trip on the known
accumulator also reproduces the grid exactly at machine precision.

**This is the first positive structural result of the phase: the non-invertibility barrier
recorded in Phase 10.3 is CLOSED for the grid-decode purpose.** It is `OBSERVED` on grids up to
9x9 (initial run) and re-measured at larger sizes below. It does **not** yet mean the operator
gap is closed: decoding a wave is not the same as compiling a task operator.

### 2.3 Decoder scale: the first sub-exact numbers were MY optimization budget

At larger sizes the gradient decoder first came out sub-exact, with mean block cosine near 0.91
(a signature of non-convergence, not of an information limit). Sweeping effort at fixed grids
separates the two:

| grid | steps | lr | exact | cell accuracy | mean block cosine |
|---|---|---|---|---|---|
| 8x8 | 400 | 0.1 | False | 0.9375 | 0.914112 |
| 8x8 | 1500 | 0.1 | True | 1.0000 | 0.988501 |
| 8x8 | 4000 | 0.1 | True | 1.0000 | 0.997861 |
| 8x8 | 4000 | 0.05 | True | 1.0000 | 0.995401 |
| 16x16 | 400 | 0.1 | False | 0.9688 | 0.909645 |
| 16x16 | 1500 | 0.1 | True | 1.0000 | 0.987430 |
| 16x16 | 4000 | 0.1 | True | 1.0000 | 0.996242 |
| 16x16 | 4000 | 0.05 | True | 1.0000 | 0.990902 |

Cell accuracy tracks the cosine toward 1.0 as effort rises, and an **exact algebraic** path
(left-inverse on the measured accumulator, no gradient descent) is exact at every size:

| grid | unknowns | grid exact | cell accuracy | max indicator err |
|---|---|---|---|---|
| 8x8 | 256 | True | 1.0000 | 5.84e-06 |
| 16x16 | 1024 | True | 1.0000 | 4.29e-06 |

So the earlier sub-exact figures were an optimization budget, recorded as a defect rather than
reported as an encoder limit. Full-sweep results at 24x24 and 32x32 remain sub-exact at the
budget tested and are retained in the receipt as unfinished, not as a bound.

## 3. Directive 2 — incumbent preserved

`arc_task_functor.py` keeps `HENRI_FUNCTOR_FIT` defaulting to the per-slot diagonal
least-squares estimator. The UWSH work is an **additional arm**, not a replacement, and the
default path is unchanged. The sealed held-out figure for the incumbent remains
+0.4215.

## 4. Zone A arithmetic: the 64-byte claim is true only at k=16

Receipt: `experiments/verification/uwsh_zone_arithmetic_observed.json`.

| quantity | value |
|---|---|
| k required for 64 bytes (16 float32) | 16 |
| basis `U` bytes at k=16 | 4194304 |

16 float32 is exactly 64 bytes, so the swarm-agent figure is **arithmetically exact at k=16**.
But the same directive also states `k <= 4..8` and `U_k` in `R^(D x 4)`, which give 16 B and
32 B — those cannot both hold with 64 B. And the frozen basis itself costs
4.0 MiB at k=16, against the
directive's "~2.5 MB" (which corresponds to k~10). The 64 B/agent figure is therefore a
**coordinate payload**, not an agent footprint.

## 5. Zone C meta-orchestrator: the "< 1 KB" budget holds as data

Counted from the controller the directive specifies (4 monitored channels: Sagnac stress,
topological-charge drift, Stiefel orthogonality, free-energy dissipation; 3 actuators:
Sagnac veto sensitivity, Langevin microheater dissipation, staticity threshold):

| quantity | value |
|---|---|
| data floor | 64 |

Under 1 KB as a **payload** budget. This excludes code, pointers and the TimescaleDB client,
so it is not a process-footprint claim, and Zone C remains **FROZEN**: no substrate actuator is
enabled by this phase.

## 6. Defects found in my own work (recorded, not hidden)

| id | defect | effect if uncaught |
|---|---|---|
| V2-SHAPE | `flat_y` returned `[NB,4]` instead of flat `[2N]` | 100/100 fits raised a broadcast error; fixed with `.reshape(-1)` plus explicit shape asserts |
| V2-RECEIPT | a 5-task smoke run wrote the canonical `_60_` receipt | a non-canonical run would have been read as canonical; canonical name now guarded |
| V3-WHITEN | per-*slot* whitening inside a per-*block* normalisation | collapsed the ceiling to ~0.002; arm removed as MY error, not reported against UWSH |
| V3-DEAD | dead `if False else None` scaffolding in the evaluator | unexecuted code paths masquerading as logic; removed |
| V4-CONV | sub-exact >9x9 decode reported too early | would have implied a false information bound; separated by an effort sweep plus an exact algebraic path |

The `gamma` baseline and the 64-byte claim are directive-side defects, not mine, and are
reported as such.

## 7. Evidence class and standing

`OBSERVED`: all figures, local CPU, `torch 2.13.0+cpu`, wall clock 12.7s (UWSH) and
33.2s (follow-up). No remote GPU was used. Zone C remains FROZEN.

Next falsifiable boundary: the operator gap is unchanged at +0.4215 held-out
against +0.7536 in-sample. UWSH is now falsified as a route, and the
oracle shows no rank-`k` restriction can recover the ceiling at constructible rank. The open
question is therefore **not** a better subspace but whether a different operator FAMILY (not a
projection of the per-slot diagonal family) is required.
