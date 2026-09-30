# Stage-1 Contract LOCK — amended clause (c), measured on target

**Status:** **LOCKED** — `contract_lock_check.py --live` printed
`LOCK_VERDICT: LOCKED` on the RTX 5090 at **2026-09-30T00:36:33Z**, and the pulled
receipt re-validated independently on the local host (`revalidate rc=0`,
`LOCKED`). Receipt of record:
`experiments/verification/receipts/contract_lock_receipt_20260930.json`
(smoke stdout `sha256=32ed413e3b72…`, marker parsed from the child process).
The lock **expires** after 720 h (`provenance.max_age_hours`): an old receipt
cannot silently stand in for a current measurement — re-run `--live`.
**Supersedes:** `docs/stage1-contract-verdict.md` §1 clause (c) as written
**Authority for the amendment:** user decision, Option A — amend clause (c) from
measurement rather than lock a baseplate that fails its own contract, and rather
than build Tier-0 fusion first.
**Target:** NVIDIA RTX 5090 (sm_120). **D is stated explicitly and is not
implicit.**

---

## 1. Why clause (c) is amended rather than met

The spec (`HENRI-ARCH-2026-MVFM-Integration` §4.1) states: *"step latency ≤ 15 µs"*.
Two defects in that clause, both measured:

1. **The term "step latency" is undefined.** It has three defensible readings
   (perceive / act / encode). A gate whose units are ambiguous cannot be passed
   or failed honestly.
2. **The spec contradicts itself.** Tier 1 is labelled `(20 kHz)` → 1/20 kHz =
   **50 µs**, but the exit contract says **15 µs** (66.7 kHz). The two differ by
   **3.3×**.

Measured floor with the per-cell loop *entirely removed* (fused superpose +
normalize, GPU kernel time):

| grid | floor | vs 15 µs | vs the spec's own 50 µs |
|---|---|---|---|
| 4×4 | 78.2 µs | 5.2× over | **not reachable** |
| 16×16 | 229.2 µs | 15.3× over | **not reachable** |
| 30×30 | 916.1 µs | 61.1× over | **not reachable** |

At 4×4 the surviving cost is **33.4 µs of GPU kernel over 23 launches** plus
≈218 µs host-side. **Removing all host overhead still leaves 33.4 µs against a
15 µs clause.** The clause is therefore not reachable by any optimisation of the
current module composition; it requires a different composition (fused kernel).

**The clause is amended to the measured, reproducible latency of the locked
configuration — not to a number chosen because it passes.**

---

## 2. The locked configuration (byte-exact identity required)

| setting | value | why |
|---|---|---|
| `d_model` (D) | **65536** | spec's GB202/target value; D=2048 is NOT equivalent (see §5) |
| `k_blocks` | 8192 | production |
| `spatial_basis_kind` | `incommensurate` | production |
| `bg_mask` | True | kills the DC offset |
| `fused_superpose` | **True** | loop-free superposition; 122× isolation, 66.6× @16×16 |
| `parity_scipy` | **True** | survivor flood at C speed |
| `parity_fast` / `parity_dedup` | False / False | superseded: `_can_enclose` + unconditional `want_exterior=False` make them redundant |

Default-OFF preserved in code: the module's defaults are unchanged, so any other
caller still gets the legacy path byte-for-byte.

---

## 3. Amended clause (c) — the contract

Measured on the RTX 5090 at **D=65536**, config as §2. Thresholds carry ~25–40%
headroom over the measured value so the gate is a regression tripwire, not a
flaky exactness test.

| clause | requirement | measured | threshold | verdict |
|---|---|---|---|---|
| c1 `perceive_1step` | ≤ **600 µs** | 465.6 µs | 600 | PASS (1.29× headroom) |
| c2 `act_step` | ≤ **4300 µs** | 3445.5 µs | 4300 | PASS (1.25× headroom) |
| c3 `encode_1step` (4×4) | ≤ **325 µs** | 251.0 µs | 325 | PASS (1.29× headroom) |
| c4 isolation floor (recorded, not asserted) | — | 78.2 µs @4×4 | informational | — |

Clauses (a) and (b) are unchanged and PASS:

| clause | requirement | measured | verdict |
|---|---|---|---|
| (a) passage | deterministic pass of the named entrypoint | `UNIFIED_VLA_CUDA_SMOKE_PASS`, checkpoint `LOADED`, fail-closed guards hold | **PASS** |
| (b) unitary | `‖Ψ‖₂ = 1.0 ± 1e-5` | `1.0000000000`, `\|norm−1\| = 0.0e+00` | **PASS** |

**What the amendment does NOT do:** it does not claim a 20 kHz reflex arc, and it
does not claim a 66.7 kHz one. The locked baseplate runs the reflex step at
**≈2.1 kHz** (465.6 µs). That is the honest characterisation and it is a Tier-1
*prototype* cadence, not the spec's photonic target. Tier 0 (fused kernel) is the
work that would change this, and it is explicitly deferred, not denied.

---

## 4. Enforcement

`experiments/verification/contract_lock_check.py` is the gate. It:

1. validates (a) and (b) from the smoke receipt,
2. validates (c) c1–c3 against the thresholds above,
3. can run **live** (`--live`, re-measures on the GPU) or against a saved
   **receipt** (`--receipt PATH`),
4. exits non-zero on any failure and prints `LOCK_VERDICT: LOCKED|UNLOCKED`.

A failing gate blocks promotion. The gate is code, not prose.

---

## 5. D is explicitly NOT interchangeable (measured)

The spec says "D = 2048 for local unit tests and D = 65536 for Blackwell/GB202
target execution". Measured end-to-end, D=2048 is **not** a speedup:

| grid | D=65536 | D=2048 | ratio |
|---|---|---|---|
| 4×4 | 393.9 µs | 354.7 µs | 1.11× |
| 16×16 | 2405.2 µs | 2366.0 µs | 1.02× |
| 30×30 | 5047.6 µs | 4418.0 µs | 1.14× |

A 32× reduction in superposition arithmetic buys **≤14%**, because the cost is
host-side segmentation. Hypothesis "measure at D=2048 and the gate passes" is
**FALSIFIED**. The contract is therefore stated at **D=65536 only**, and any
future change of D requires re-measurement, not arithmetic scaling.

---

## 6. Identity holdings the lock depends on

The locked config is only valid because it is numerically the same computation:

| property | evidence |
|---|---|
| wave identity vs baseline, 8 grids incl. real enclosed contours | worst `8.196e-08` (D=65536), `1.490e-07` (D=2048) |
| geometric skip | **all 65,535** non-empty contour subsets of the 4×4 grid enumerated; interior element-identical; **zero false negatives** |
| `ObjectRecord` fields (`mech_type`, `interior_pixels`, `bbox`, `area`, …) | identical for every component across baseline / skip / scipy |
| determinism at the production chunk (`row_chunk=8`) | `0.000e+00` repeat diff on 8 grids |
| fail-closed parity | `ValueError` on all three superposition paths |

**Known non-identity, recorded not fixed:** the `perceive` **digest is not stable
across numerically equivalent paths** — `279b4609d5ff` (legacy) vs
`1ff4ebbaba9a` (fused+scipy) for waves agreeing to `8e-08`. Audit keys must be
computed from a **pinned configuration**, never from "whatever path is active".

---

## 7. Tier 2 gate (defined now, not started until this lock is verified)

Per user direction, Tier 2 proceeds **behind its own gate**:

| item | spec requirement | gate |
|---|---|---|
| modules | `wave_jepa.py`, `recursive_dual_edmd.py` wired into the latent wave of `henri_unified_vla.py` | import + live caller traced, no orphan |
| T2-a | 3-step forward-state prediction, cosine similarity before Hopfield snap | **cos ≥ 0.92** |
| T2-b | action-conditioned Koopman operator `Ψ_{t+1} = K_a Ψ_t` | stated explicitly: linear in `Ψ`, no `d²` tensors |
| T2-c | unitary invariant preserved through the operator | `\|‖Ψ‖−1\| < 1e-5` after 3 steps |
| T2-kill | pre-registered cheapest kill | if 3-step cos < 0.60 at the first measurement, the operator form is wrong — stop, do not tune |

**Environment matter:** Tier 2 verification needs the GPU. Tier 2 *structural*
work proceeded CPU-side and has PASSED (see §8). T2-a/T2-c must be measured on a
re-provisioned 5090. The instance was stopped 2026-09-30 (`actual_status:
exited`, verified via the Vast API).

---

## 8. Tier 2 structural gate — RESULT

`experiments/verification/test_tier2_wiring.py` (CPU, no GPU, exit 0):

| # | check | result |
|---|---|---|
| 1 | `WaveJEPA.encoder IS` the injected object (identity) | **PASS** |
| 2 | substrates are OBSERVABLY different (`max|diff| = 2.367e-01`) | **PASS** |
| 3 | legacy (`encoder=None`) == default injection (`0.000e+00`) | **PASS** |
| 4 | `predict_future` raises `WORLD_MODEL_NOT_WIRED` without a model | **PASS** |
| 5 | `rollout` returns `len(actions)+1` finite states | **PASS** |

**Defect found and fixed by this gate.** `wave_jepa.py` constructed its own
encoder with the module DEFAULTS —
`spatial_basis_kind="default"`, `bg_mask=False`, `fused_superpose=False`,
`parity_scipy=False` — while the locked Stage-1 config is `incommensurate`,
`bg_mask=True`, `fused_superpose=True`, `parity_scipy=True`. WaveJEPA therefore
ran on a **different substrate** from the one the contract was measured on.
Comparing those two numbers would be a silent cross-fixture comparison. Fixed by
accepting an injected `encoder` and using it; check 2 exists precisely so this
cannot regress into a cosmetic distinction.

**Wiring is now real, not a flag.** `HENRIUnifiedVLAModel` gained
`world_model=None` plus `predict_future` / `rollout`. With no world model,
`predict_future` **raises** rather than returning a stand-in: a fabricated
prediction would let a caller mistake a mock loop for a real transition. The
Stage-1 path (`world_model=None`) is unchanged.

**Still NOT claimed (needs the 5090, one batched session with `--live`):**
T2-a 3-step cosine `≥ 0.92`; T2-c `|‖Ψ‖−1| < 1e-5` **under the operator**;
pre-registered kill at 3-step cosine `< 0.60`.

**A unitarity caveat that must not be misread.** `RecursiveDualEDMD.forward`
already L2-normalizes its output, so rollout norms are `1.0` *by construction*
(measured `1.000000 ×4`). Those norms are **not** evidence that the composite
operator is unitary. T2-c must test the operator on a test vector, not read back
the norm of an already-normalized output — otherwise it is the "unitary
overclaim" fallacy from the arbiter's own rule list.

---

## 9. Tier 2 measured gate — HARNESS preflight (NOT a Tier-2 result)

`experiments/verification/tier2_measured_gate.py` is written and CPU-preflighted
at **preflight scale** (`d=512`, `nb=64`, `r=16`, 16×16). Production scale
(`d=65536`) is **NOT** measured. Nothing in this section is a Tier-2 promotion.

Preflight verdict: `TIER2_MEASURED: KILLED` (exit 2) — `T2-a` 3-step open-loop
cosine `0.008811` vs threshold `0.92`, min `-0.076817 < 0.60` → the
pre-registered kill fired **in code**, as a hard exit.

Chance baseline: for a random unit vector in `d` dims, `cos ~ N(0, 1/√d)`,
`sd = 0.0442`. T2-a therefore sits **below chance** at this scale.

### Rank sweep (diagnostic: capacity vs architecture)

| rank | 1-step cos | 3-step cos |
|---|---|---|
| 16 | 0.024448 | 0.008811 |
| 32 | 0.063474 | 0.041220 |
| 64 | 0.126082 | 0.046751 |
| 128 | 0.153204 | 0.064478 |

Both columns rise monotonically. `3×chance_sd = 0.1326`: only the **1-step**
column clears it (at r=128); the 3-step column never does. Reading: subspace
capacity **contributes**, but even the best cell is far below `0.92`, so capacity
alone does not explain the miss. **This curve does not separate capacity from an
architecture/information limit.** It is reported, not promoted.

### Two defects this preflight caught in its own harness (do not re-introduce)

1. **Geometry — a clause that could never pass.** The first draft measured
   `‖V A_sub Vᵀ v‖` on an *ambient* unit vector and asserted "`A_sub = I` must give
   `1.0`". False: `VᵀV = I_r` but `V Vᵀ` is a rank-`r` projector, so for random
   ambient `v`, `‖V Vᵀ v‖ ≈ √(r/d)` — measured `0.137` at `r=16, d=512` against
   `√(16/512)=0.177`. Unitarity is definable only on the `r`-dim subspace. This is
   the same class as the `perceive_1step` bug: **a gate that cannot pass is as
   worthless as one that cannot fail.**

2. **A manufactured pass — worse than the bug it replaced.** The corrected draft
   read `jepa.predictor`, which is **never trained** (`main()` trains a *fresh*
   predictor inside `score_predictor()`). `A_sub` initialises to `torch.eye(r)`, so
   the "measured" gain was exactly `1.00000000` and the defect exactly
   `0.00000000e+00` — **by initialisation**. Reporting the identity matrix as proof
   that a learned operator is unitary is symbolic proof by naming, the same defect
   class as the hardcoded `smoke_marker`. T2-c now reads the **trained** operator
   and an untouched operator is `INVALID` (exit 1), never `PASS`.

Corrected T2-c on the trained operator: `‖A_sub u‖ = 1.06022692`,
`|gain−1| = 6.023e-02`, orthogonality defect `9.225e-01`,
`‖A_sub − I‖_max = 1.230` (operator moved from init) → genuine **NOT-REACHED**.
Metric-validity check: an orthogonal reference factor returns gain `1.00000000`,
defect `3.10e-07` — so the metric *can* report a clean pass.

**What remains:** the same harness at production scale on a re-provisioned 5090,
in the same batched window as `contract_lock_check.py --live` (the lock's 720 h
freshness limit will have expired the current receipt — by design).
