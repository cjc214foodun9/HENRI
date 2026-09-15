# Phase 10.1 — Operator Gap Adjudication & Sample-Efficiency Directive

**Provenance (OBSERVED, authenticated locally 2026-09-15).**

| Field | Value |
|---|---|
| File | `Project HENRI_ Phase 10.1 Operator Gap Adjudication & Sample-Efficiency Directive.pdf` |
| SHA-256 | `972c29ffdc67d125ff54009850976fa107d8fb9481dcb42bc70363df66ce0fd9` |
| Pages | 11 |
| Bytes | 521,330 |
| Extracted chars | 16,893 |
| Duplicate of prior docs? | **No** (`779d39f6…`, `7654f02c…`, `3335bb8b…` all differ) |

Distrust note: advisory reference blocks in the session that cited this document
reported sha256 `1b13e3b9…`, **19 pages**, ~39,585 chars, and a gate value of
`<= 0.28`. None of those match the authenticated extraction. That material was
`FALSIFIED` and discarded. Only the values below were used.

## 1. Adjudications (both confirmed by the measurement record)

| Decision | Order | Basis |
|---|---|---|
| `progressive_semantic_grounding_engine.py` | **FREEZE, do not patch.** Fail-closed assertion. | It sits behind `HENRI_ARC_PSG` (default-OFF). Editing dormant code that does not move the active baseline spends velocity without touching the binding constraint. |
| Vast instance `50797414` | **Stay `exited`, zero funding.** | The 60-task split runs local CPU `torch 2.13.0+cpu` in **3.3 s** with float32 parity (Δ 3.2e-07 vs the GPU receipt). The constraint is algebraic, not compute-bound. |

## 2. Root cause, and the correct scope of the cancellation theorem

The Phase 10 receipt showed the `0.3321` gap is invariant across seven orders of
magnitude of ridge (`1e-9 → 1e+1`, closure at `1e-4` = `6.9e-05` = 0.021% of the
gap). The directive's explanation is that a **per-slot diagonal** operator is
translation-blind by construction:

```text
(T X)* (T Y) = X* T* T Y = X* Y      for unit-modulus diagonal T
sum |T X|^2  = sum |X|^2
```

Measured on the live encoder (`koopman_bank_feasibility_probe.py`, CPU,
`roll_multiplier`):

- `|T|` is unit modulus to `max_dev 6e-08` for all tested `(dw, dh)`.
- `max|conj(TX)·TY − conj(X)·Y| = 1.2e-06 … 1.9e-06` (float32 floor).
- `rel |sum|TX|² − sum|X|²| = 0.0`.
- **Exact law:** orbit augmentation over the translation group satisfies
  `W*_orbit(λ) ≡ W*_plain(λ/S²)`. It is a **ridge rescale**, not new information.

**Scope correction.** The theorem rules out *translation and ridge* remedies. It
does **not** establish that the gap is caused by non-diagonality — the premises
"ground truth is translation-invariant" and "augmentation is the estimator's only
route" are the directive's, not measured facts. The legitimate consequence is
narrower and still decisive: the diagonal family is translation-blind, so
**translation augmentation and ridge tuning cannot close this gap**, and
augmentation that *does* change `W*` changes it only through the ridge.

## 3. The K ≤ 8 generator bank — enumeration discrepancy, recorded not invented

The directive's section 5 header says `K <= 8` but enumerates exactly **six**
generators. `6 ≠ 8`. The two "missing" entries were **not** invented; the
resolution used here is recorded explicitly:

| # | Generator | Realization from live buffers | Diagonal in freq basis |
|---|---|---|---|
| 1 | Identity pass-through | `L = I` | yes |
| 2 | Spatial differential currents | split: `i·wx`, `i·wy` (how the operators are actually spanned) | yes |
| 3 | Dihedral reflection parity | split: `(kx,ky)→(-kx,ky)`, `→(kx,-ky)` as **slot permutations** | **no** |
| 4 | Laplacian diffusion | `-(wx²+wy²)` | yes |
| 5 | Color permutation shift | uniform phase advance `exp(iθ)` | yes |
| 6 | Topological charge projection | mask over the reserved DC slots | yes |
| 7 | *(added)* per-block `SL×SL` unitary mixer | `torch.linalg.qr` on live `BLOCK_SLOTS=4` | **no** |

Splitting 2 and 3 is the honest way to span the named operators; the seventh
entry reaches `K=8` and is labelled `ADDED_` so no analysis can mistake it for a
directive-named generator.

### Shape constraints measured, not assumed

- A dense bank `[K, D, D]` complex64 at `K=8, D=32768` is **68.7 GB**. The bank
  is therefore a list of **callables** `L_j(X) -> X'`, never matrices.
- The directive's sketch passes `generators: [K, D]` **diagonal masks**. Any
  `W* = Σ α_j L_j` over diagonal `L_j` is diagonal — i.e. inside the family the
  directive's own section 4.2 calls insufficient. The sketch cannot satisfy its
  own requirement; coupling requires callable or dense `[D, D]` operators.
- `BLOCK_SLOTS = 4` and the flat complex index **is** the `(block, slot)` index
  space (`D/2 = NB·4 = 32768`).
- `(kx,ky)→(-kx,ky)` target closure **within a block is only 0.2759** (and
  `0.2751` for `-ky`). Frequencies are drawn randomly per block, so a dihedral
  prior is **not** a within-block permutation — it must act across the whole bank.

## 4. Acceptance gate (recovered from pdf p.8)

The request text lost these to a math-mode failure. Recovered verbatim:

```text
Target:    held-out score must exceed 0.5000 on the 60-task split.
Condition: gap must contract by at least 25% (Delta <= 0.2490) relative to ceiling.
```

Self-consistency check: `0.3321 × 0.75 = 0.24908` ≈ `0.2490`. ✓

## 5. Freeze semantics implemented

The directive prints a **module-level** raise. That form breaks
`experiments/performance/psg_cuda_check.py`, which sets `HENRI_ARC_PSG="1"` at its
line 24 and imports the engine at line 26 — the import itself would raise before a
typed status could be reported. The freeze is therefore enforced at the two
**compute entry points** named in the directive:

- `compile_functor_wave` (the `conj(wx)*wy` site)
- `ProgressiveSemanticGroundingEngine.compile_task_functor` (the delegating site)

Requested path = `HENRI_ARC_PSG == "1"`; override = `HENRI_ZONE_C_FREEZE == "0"`
(explicit, for development). Raise is `RuntimeError("BLOCKED_ZONE_C_FROZEN …")`.
With the flag OFF the engine is inert, so the existing contract tests stay green.
The freeze does **not** modify either naive operator; a contract test asserts both
sites are byte-identical, so "frozen" cannot silently become "refactored".

## 6. Honest limits

- The metric is internal-representation held-out **cosine recovery**, not an ARC
  solve rate. No external benchmark claim.
- AAII ladder: not started, by design.
- Zone C stays at the two verified parameter-free gates (charge `Δq = 0`, torus
  equivariance conditional on `H = W = S`). The generator bank is an **operator
  basis**, not Zone C content, and writes nothing to `zone_c_*`.
- No TimescaleDB DDL was executed; under the freeze directive, building a storage
  layer for a frozen subsystem would be fabrication.

## 7. MEASURED OUTCOME (OBSERVED, local CPU, 60 tasks, `n_skipped=0`, 5.3 s)

`experiments/verification/evaluate_60_task_koopman_gap.py` →
`evaluate_60_task_koopman_gap_observed.json`. `torch 2.13.0+cpu`.
Baseline replica **reproduces** (held `0.4215` / ceiling `0.7536`), so the
comparison is valid and the run is not VOID.

| arm | held-out | ceiling | gap | frac beating identity |
|---|---|---|---|---|
| identity | **+0.4033** | — | — | — |
| `legacy` (mean conj-corr) | +0.4071 | +0.6994 | 0.2924 | 40.0 % |
| `diag_ls` (incumbent) | **+0.4215** | **+0.7536** | **0.3320** | 45.0 % |
| `koopman_8@0.01` | +0.3907 | +0.4180 | 0.0274 | 30.0 % |
| `koopman_8@0.1` | +0.3661 | +0.3922 | 0.0261 | 13.3 % |
| `koopman_diagonly@0.01` | +0.3879 | +0.4152 | 0.0273 | 30.0 % |
| `koopman_diagonly@0.1` | +0.3634 | +0.3892 | 0.0258 | 13.3 % |
| `koopman_9_withmixer@0.01` | +0.3917 | +0.4191 | 0.0274 | 31.7 % |
| `koopman_9_withmixer@0.1` | +0.3674 | +0.3937 | 0.0263 | 13.3 % |

### The acceptance gate FALSIFIED

| gate | required | measured | pass |
|---|---|---|---|
| held-out | `> 0.5000` | **`0.3917`** (best Koopman) | **NO** |
| gap | `<= 0.2490` | `0.0274` | yes — **but see below** |

`ACCEPT: false`. Koopman does **not** beat `diag_ls` (`0.3917 < 0.4215`) and does
**not** beat identity (`0.3917 < 0.4033`). The K ≤ 8 subspace, with the
directive's named generator set, is **insufficient** for this representation.

### The "gap contraction" pass is a vacuous metric — do not report it as progress

The gap fell `0.3320 → 0.0274` because the **in-sample ceiling collapsed**
`0.7536 → 0.4180` (−0.336), while held-out moved only `0.4215 → 0.3917`
(**−0.030, i.e. it got worse**). The gap is `ceiling − held_out`, so **any strictly
weaker operator shrinks it**. A metric that improves when the model degrades is
not evidence. The absolute held-out floor is what caught this, and it is the only
gate component that should carry weight here.

### Ablations: mode coupling contributes nothing

- `koopman_8` → `koopman_diagonly` (removes BOTH dihedral permutations, the only
  genuinely off-diagonal operators present): held-out `0.3907 → 0.3879`, `−0.0028`.
- `koopman_8` → `koopman_9_withmixer` (ADDS one off-diagonal `SL×SL` unitary):
  held-out `0.3907 → 0.3917`, `+0.0010`.
- An earlier arm called `koopman_named6` was **deleted**: in the phase-split bank it
  filtered to the same 8 generators as `koopman_8`, so it was a duplicate by
  construction, not an ablation. Its matching numbers were **not** evidence.

So the directive's section 4.2 requirement ("only an operator family that couples
distinct frequency modes can cross the floor") is **not satisfied by this bank**,
and — more usefully — the off-diagonal directions that *are* present buy ~0.
The subspace, not the coupling, is the constraint.

### Why the subspace is too small: rank deficiency, measured

`mean_rank_G = 5.667` of `K = 8` across 360 solves; `n_gram_effectively_diagonal = 0`.
Six of the eight generators (identity, `i·wx`, `i·wy`, Laplacian, colour shift,
charge projection) are **diagonal** in the live frequency basis and therefore
span a space the diagonal family already covers; the two reflections are
permutations. An 8-dimensional hypothesis class cannot represent a spatially
non-local task operator, so restricting D=32768 → K=8 discards exactly the
capacity the task needs. The directive's sample-efficiency argument (8 params vs
32768) is sound **only if the truth lies in the span** — and this bank shows it
does not.

### Two defects of my own found and fixed, both of which changed the result

1. **Reflection was a collapsing projection, not a permutation.** v1 mapped each
   slot to the first slot anywhere in the bank holding the reflected frequency;
   since all `S² = 1024` frequencies occur many times among 32768 slots, ~32768
   slots collapsed onto ~1024 targets. Fixed to a genuine involution
   (`dst[dst] == src` asserted at build). This *lowered* the Koopman arms
   (`0.4113 → 0.3907`) — the earlier, higher number was flattered by my bug.
2. **The ridge was numerically absent.** `reg_lambda = 1e-2` read as an absolute
   value against a mean Gram diagonal mass of `2.534e+07` is a relative `4e-10`;
   `1e-2` and `1e-1` returned bit-identical results. Now relative by default
   (`lambda_eff = reg_lambda * mean|diag(G)|`), which is why the two lambdas now
   differ. Solving a rank-5.75-of-8 system without an effective ridge is the exact
   degeneracy a ridge exists to remove, so the first pass did **not** fairly test
   the hypothesis.

This is the **absolute-versus-relative** defect class for the fourth time in this
line of work. It is now recorded as a standing hazard in
`scale-conflation-and-device-traps.md`.

### What would actually be needed (not attempted here)

A bank that spans the needed operators, chosen from **measured** task structure
rather than from a physics-flavoured list — e.g. generators fit as low-rank
corrections to the diagonal operator on a *training* split, then frozen. That adds
parameters and so trades away the very sample efficiency the directive wanted. The
tension is the real finding: at `M = 3` demonstrations the estimate is
underdetermined (3 samples, 32768 slots) and the subspace is over-restrictive
(8 parameters for non-local structure) **at the same time**. The binding
constraint is therefore not the operator family but the **effective demonstration
count**, which translation augmentation cannot raise (section 2).

