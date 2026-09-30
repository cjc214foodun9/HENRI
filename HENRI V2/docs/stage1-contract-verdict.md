# Stage-1 Contract Verdict — the minimal viable reflex loop, measured

> **SUPERSEDED ON CLAUSE (c):** the clause-(c) verdict below (FAIL 31.0x / 229.7x
> / 16.7x against the spec's literal 15 us) stands as recorded evidence. The
> *resolution* of that failure is the amended contract in
> `docs/stage1-contract-lock.md`, which is **UNLOCKED** until
> `contract_lock_check.py --live` prints LOCKED from a measured receipt. Read the
> two together; do not read this file's FAIL as a live gate state.

**Date of measurement:** 2026-09-29 / 2026-09-30
**Target:** NVIDIA RTX 5090 (sm_120, 33.67 GB), `torch 2.12.0+cu130`, scipy 1.18.0
**Entrypoint:** `experiments/verification/smoke_unified_vla_cuda.py`
**Method:** every number below comes from `experiments/verification/substrate_e2e.py`
and `experiments/verification/fused_probe.py` executed on the RTX 5090, with the
two production modules transferred by SHA256-verified copy (the instance's git
remote points at a stale local bundle, so `git fetch` there returns old commits).

---

## 1. What the contract is

The spec (`HENRI-ARCH-2026-MVFM-Integration` §4.1) states one exit gate with
three clauses. Clause (c) is undefined as written, so this document defines it as
three separate measurements and reports all three. **Ambiguity in a gate is a
defect in the gate**, not a licence to pick whichever reading passes.

| clause | requirement | source |
|---|---|---|
| (a) passage | deterministic pass of the named entrypoint | spec §4.1 |
| (b) unitary | `‖Ψ‖₂ = 1.0 ± 1e-5` | spec §4.1, §3.2 |
| (c) latency | "step latency ≤ 15 µs" — **undefined term** | spec §4.1 |

Three defensible readings of (c), all measured:

| reading | definition | measured |
|---|---|---|
| c1 `perceive_1step` | one `vla.perceive(grid)` | **465.6 µs** (fused) / 707.1 µs (stock) |
| c2 `act_step` | one `vla.act(...)` | **3445.5 µs** (fused) / 3436.7 µs (stock) |
| c3 `encode_1step` | one `encode_grid(grid)` at 4×4 | **251.0 µs** (fused) / 583.8 µs (stock) |

---

## 2. Verdict

| clause | verdict |
|---|---|
| (a) passage | **PASS** — `UNIFIED_VLA_CUDA_SMOKE_PASS`, checkpoint `LOADED`, fail-closed guards hold |
| (b) unitary | **PASS** — `‖Ψ‖ = 1.0000000000`, `|norm−1| = 0.0e+00`, finite |
| (c) latency | **FAIL** — 31.0× over the 15 µs clause (c1), 229.7× (c2), 16.7× (c3) |

**The minimal loop is NOT locked.** Clause (c) does not pass and is not claimed
to pass.

---

## 3. End-to-end encode latency (D=65536, construction excluded)

| grid | baseline | fused+scipy | speedup | kernel µs | launches |
|---|---|---|---|---|---|
| 4×4 | 583.8 | **251.0** | 1.49× | 33.4 | 61 → 23 |
| 16×16 | 10685.5 | **2230.4** | 4.60× | 194.6 | 1671 → 28 |
| 30×30 | 97629.9 | **4922.9** | 19.34× | 858.0 | 5859 → 38 |

**Probe defect, disclosed.** A first version of this probe constructed the
encoder *inside* the timed lambda (`mk(D, **kw).encode_grid(g)`), so its absolute
numbers include encoder construction. PART 3 constructs once and times only
`encode_grid`; those are the clean numbers quoted above. The baseline column is
from the PART 1 run and is therefore inflated by the same constant
(≈ 143 µs at 4×4), so the true speedups are slightly LARGER than shown. The
ratios are directionally correct; the absolute baseline should be re-measured
without construction before being quoted as a baseline.

## 4. The full reflex arc

| config | perceive | act | `‖Ψ‖` |
|---|---|---|---|
| stock | 707.1 µs | 3436.7 µs | 1.0000000000 |
| fused+scipy | **465.6 µs** | 3445.5 µs | 1.0000000000 |

`act` is **unchanged** (3436.7 → 3445.5). The encoder fixes do not help the act
path, which is dominated by the swarm orchestrator and the EFE planner
(`num_experts=1024, r_rank=16`). Attributing act latency to the encoder would be
a category error.

---

## 5. Two hypotheses killed by measurement

**(i) "Measure at D=2048 and the gate passes."** FALSIFIED. D=2048 changes
end-to-end encode latency by 1.02–1.14×:

| grid | D=65536 | D=2048 | ratio |
|---|---|---|---|
| 4×4 | 393.9 | 354.7 | 1.11× |
| 16×16 | 2405.2 | 2366.0 | 1.02× |
| 30×30 | 5047.6 | 4418.0 | 1.14× |

The encoder is **not** superposition-bound, so re-quoting clause (c) at the
smaller D does not rescue it. The 32× reduction in superposition arithmetic buys
at most 14%.

**(ii) "Kernel launch count is the cost."** FALSIFIED earlier at 78× fewer
launches for 1.09× wall time. Confirmed here: the fused path cuts launches
61→23 (4×4) and 1671→28 (16×16), and the wall gain comes from removing the
Python loop, not from launch reduction per se.

---

## 6. The measured floor, and why 15 µs is out of reach

Best achievable by removing the per-cell loop entirely (fused superpose +
normalize, GPU kernel time):

| grid | floor | vs 15 µs | vs 50 µs |
|---|---|---|---|
| 4×4 | 78.2 µs | 5.2× over | NOT reachable |
| 16×16 | 229.2 µs | 15.3× over | NOT reachable |
| 30×30 | 916.1 µs | 61.1× over | NOT reachable |

At 4×4 the surviving kernel time is **33.4 µs with 23 launches**, and ≈ 218 µs
of the 251 µs total is host-side (grid→device 8.5, device→numpy 7.0,
segmentation, parity build, dispatch). Removing ALL host overhead still leaves
33.4 µs of GPU work against a 15 µs clause.

**The spec contradicts itself.** Tier 1 is labelled "(20 kHz)" → 1/20 kHz =
50 µs, but its exit contract says "≤ 15 µs" (66.7 kHz). The two differ by 3.3×.
Even the spec's own tier label is not met by the measured floor at 4×4 (78.2 µs).

---

## 7. Identity of every optimisation

Worst absolute wave difference vs the unmodified baseline, 8 grids including real
enclosed contours (rings, nested shapes):

| D | worst diff | config |
|---|---|---|
| 65536 | 8.196e-08 | 30×30 / fused |
| 2048 | 1.490e-07 | 30×30 / fused |

All six configurations (baseline / fast / scipy / fused / fused+scipy /
fused+fast) agree within 1.5e-07. Segmenter proofs: all **65,535 non-empty
contour subsets** of the 4×4 grid enumerated — interior element-identical,
**zero false negatives** from the geometric-skip predicate, `mech_type` and every
other `ObjectRecord` field identical.

**Caveat — the digest is not stable across equivalent paths.** `perceive` returned
digest `279b4609d5ff` (stock) and `1ff4ebbaba9a` (fused+scipy) for a *wave that
agrees to 8e-08*. Any audit key built on the digest must therefore be computed
from a pinned configuration, not "whatever path is active". Flagged, not fixed.

---

## 8. What is NOT claimed

- No score claim. No benchmark result. No task outcome.
- Clause (c) is **not** met and is not in any way implied to be met.
- `fused_superpose` is **default-OFF**; the default path is byte-identical to
  the pre-existing code except for the two geometric-skip fixes, which are proven
  identical by exhaustive enumeration.
- No Tier 2/3/4 work has started, and none should start while clause (c) fails.

---

## 9. Decision required

Two honest options, both with measured evidence:

**Option A — amend clause (c) from measurement.** Replace "≤ 15 µs" with the
measured floor for this composition, state D explicitly, and lock clauses (a)+(b)
plus the amended (c). This locks a baseplate that genuinely passes its own
contract. Cost: the amended number is 17–31× the original, so it must be
justified as a Tier-1 tier (not a 20 kHz reflex arc) in the document.

**Option B — Tier-0 fused kernel.** Write the segmentation + parity + superpose +
normalize path as a Triton kernel (or one CUDA graph over the pure-tensor
region), targeting the 33.4 µs of measured GPU work. This is a real project, not
a micro-optimisation, and the CPU segmentation that dominates 16×16/30×30 must
move to the device for it to matter.

Neither option should be taken by assumption. The baseplate is not locked.
