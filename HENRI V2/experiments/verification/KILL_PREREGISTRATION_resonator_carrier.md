# KILL PRE-REGISTRATION — Tripartite VSA Resonator Carrier

**Status:** `PRE-REGISTERED` — written and committed **before** the carrier exists.
**Carrier under test:** `HENRI V2/henri_resonator.py` (proposed, default-OFF).
**Gate tests:** `HENRI V2/tests/contract/test_resonator_kill_gates.py`
**Base:** merge of `main` (`85149cf`) into `origin/carrier/e6-physical-verifier` (`3ae27c7`).
**Date:** 2026-09-15 · **Author:** HENRI arbiter (Root Holon)
**Environment:** CPU only. `cuda_available=False`; Vast `50797414` EXITED unfunded; prod Zone C `:10100` closed.
**Evidence class ceiling for all results below:** `ENGAGED_WIRING_ONLY` — wiring and invariants, never capability.

## Why this document exists before the code

The supplied blueprint (`HENRI-ARCH-2026-METAMATERIAL-RECURSIVE-SYNTHESIS`, SHA-256
`2ee7005f525ded086405ffea82b122111c26377fd2241587cdd39599c357d4e7`) was `FALSIFIED`:
its spatial branch was content-invariant, its asserted shift `(2,-1)` ranked 30/49 against
its own objective, and its Sagnac gate passed an all-zero field. A rebuild must not repeat
those defects. Pre-registering the kill conditions **before** writing the mechanism prevents
post-hoc threshold drift — the failure mode where a mechanism is judged "working" because
its criteria were chosen after seeing its output.

## Measured starting facts (OBSERVED, live tree, this session)

| Fact | Measured value | Consequence for the carrier |
|---|---|---|
| `TorusIngressEncoder.encode` content-dependence, two different grids | `\|Δ\|_inf = 1.975276` | Live encoder is **not** content-invariant; the packet's core defect is absent live |
| `encode` determinism, same grid twice | `0.000000` | Safe to use as a stable reference |
| `apply_roll` exactness, shifts (1,0)(0,1)(3,0)(3,-2)(7,5) | rel. err `2.7e-04 … 3.2e-04` | Roll is a valid **diagonal** factor; noise floor ≈ 3e-4 |
| Colour change 3→5, `z_B / z_A` | `\|ratio\| mean 5.2552, std 37.94`, `angle std 0.928` | Colour is **NOT** a per-slot diagonal op — no rotor diagonalization may be assumed |
| Enclosure, solid vs ring, `z_B / z_A` | `\|ratio\| mean 2193.7, std 14975.8` | Mask is **NOT** diagonal — must be searched, not solved |
| `evaluate_veto(cand=X, axiom=X, world=Y)` | `(0.0, 1.0, False, 'SAGNAC_VETO_OK')` | Live gate contract: `(delta, coherence, vetoed, status)` |
| `evaluate_veto` on **all-zeros** | `(0.5, 0.5, True, …)` | **Live gate FIRES on a dead field** — it passes the control the packet failed |
| `store_engrams([1,8,8])` | **accepted**, `engrams.shape=(1,8,8)` | `ContinuousHopfieldCleanup` does **not** validate rank (defect). Carrier must flatten explicitly |
| `lexical_snap([1,D], top_k=2)` | `idx [[0,1]] conf [1.0, -0.0497]` | Codebook egress works when `dim == wave.shape[-1]` |

**Design consequence.** Only the **roll** factor is provably diagonal. The carrier therefore
searches all three factors over explicit codebooks with a clean-up step each iteration. It does
**not** solve for colour or mask analytically, because the measurement above forbids that
assumption.

## The identity-attractor defect this pre-registration defends against

A factorization search whose objective is minimized at the identity transformation is
indistinguishable from "do nothing". The falsified packet exhibited exactly this: its objective
collapsed to `mean(|T - 1|)`, globally minimized at `(dx,dy) = (0,0)`.

**Therefore every K2 task below uses a ground-truth transformation that is NOT the identity**,
and the identity rate is reported alongside the hit rate. A carrier that "passes" only by
returning identity is a failure, and the report must say so.

---

## K1 — Fibre binding carries position, and enclosure is separable

**K1a · Content-dependence.** Two grids differing only in object position must produce waves
separated above the noise floor.
- Threshold: `\|enc(g1) − enc(g2)\|_inf ≥ 1e-3`.
- Predicted: PASS (measured 1.975 on the live encoder).
- **NEGATIVE CONTROL (must FAIL):** a content-blind branch that computes its spatial phasor
  from index buffers only — the packet's `psi_pos` — must **fail** K1a. If the control passes,
  the test is not measuring what it claims and K1 is void.

**K1b · Determinism.** Same grid, twice, exactly equal. Threshold: `== 0.0`.

**K1c · Position separation via the roll operator.** For a known roll `(dw,dh) ≠ (0,0)`, the
operator residual at the true shift must beat the median residual over wrong shifts by a
declared margin.
- Threshold: `residual(true) ≤ 0.5 × median(residual(wrong))`.
- Predicted: PASS for the roll factor (roll is diagonal and measured exact to 3e-4).

**K1d · Enclosure separability.** A solid block and a hollow ring of equal mass must be
distinguishable by a declared topological feature.
- Threshold: feature distance `≥ 1e-6` **and** the feature is not a monotone function of mass.
- Predicted: PASS via `ParityContourMask.compute_parity_contour` interior/exterior split.
- Reject if the feature is `area` (mass), which cannot separate solid from hollow.

**K1 verdict:** K1 fails if any of a/b/c/d fails, or if the K1a negative control passes.

---

## K2 — The resonator performs a real search

**Task.** Synthesize a target wave by applying a **known composite** transformation
`T = roll(dw,dh) ⊙ rotor(colour) ⊙ mask(interior/boundary)` with `(dw,dh) ≠ (0,0)`.
The composite is **not** rank-1 in the factor codebooks.

**Requirement.** On a **disjoint** split, the resonator's recovered factor triple must beat
the identity baseline.

- Primary metric: `hit_rate` = fraction of trials recovering all three true factor indices.
- Required: `hit_rate ≥ 0.60` **and** `hit_rate − identity_rate ≥ 0.30`.
- `identity_rate` = fraction of trials where the identity triple `(roll=(0,0), rotor=id, mask=all-ones)`
  is returned. **Reported unconditionally.**
- Secondary: converged residual must be strictly below the first-iteration residual on
  `≥ 80%` of trials; iteration count reported; clean-up hit rate per factor reported.

**Pre-registered expected defect.** If the objective is minimized at identity, `hit_rate` will
collapse toward `identity_rate` and the margin test will FAIL. That outcome is a **negative
result and a governance win**, not a formatting problem. It must be reported as-is.

**K2 verdict:** FAIL if `hit_rate < 0.60` or the margin `< 0.30`, or if any trial returns
identity while claiming a non-identity target.

---

## K3 — The Sagnac gate discriminates a dead field from a conserved one

**Mandatory negative control.** An all-zeros field and a constant field **must fire the veto**.
- Measured on the live gate: all-zeros → `vetoed=True`; constant → `vetoed=True`. PASS.
- The falsified packet's gate returned `delta_q = 0` for the same dead fields → `VETO=False`.
  That is the defect K3 exists to exclude.

**Positive control (must NOT fire).** A candidate equal to the axiom with a conserved world
field must return `vetoed=False`.
- Measured: `evaluate_veto(X, X, Y) → (0.0, 1.0, False, 'SAGNAC_VETO_OK')`. PASS.

**Scope statement (required in the report).** `arc_sagnac_veto.evaluate_veto` is **advisory**:
it re-ranks EFE candidates and never replaces `EFEPlanner.select_action`. The carrier must not
promote it to a hard selector. Reuse it; do not build a second gate.

**K3 verdict:** FAIL if a dead field does not fire, or if the conserved positive control fires.

---

## K4 — Zone C artifact DAG is fail-closed

**Additive migration only.** No column drop, no `NOT NULL` relaxation, no backfill of
`legacy_unattributed` rows. Ledger advances to **v4**.

**Write-time fail-closed.** A persistent parent-child write missing any of
`run_id`/`arm_id`/`commit_sha`/`domain_family` must **raise**, not inherit a default.
- Negative control: write with `commit_sha=None` **must fail**.
- Round-trip: parent → child → read-back returns identical lineage fields.

**K4 verdict:** FAIL if any incomplete write is silently accepted, or if a pre-existing row's
lineage is mutated.

---

## Handling of the three supplied papers and the HF archive

All three arXiv IDs were authenticated from primary bytes (HTTP 200, one `<entry>` each):
`2512.24695` (Nested Learning), `2506.21734` (HRM v3), `2512.24601` (Recursive Language Models v3).

**Pre-registered claim ceiling.** None of these is a training-time substitute for the HENRI
wave substrate, and **no architectural equivalence is claimed**:

| Source | Its actual mechanism | How it enters the carrier | What is NOT claimed |
|---|---|---|---|
| Nested Learning (Google) | Multi-level **optimization**; optimizers as associative memories | Framing only: the clean-up memory is documented as a one-level associative memory inside a multi-level loop | No optimizer is replaced; no equivalence to Hope/continuum memory |
| HRM (Sapient) | **Trained** 27M-param dual-timescale recurrent net | Structural analog: slow/high-level factor estimate + fast/low-level refinement inside the resonator loop | Its ARC result is a **trained** result and does **not** transfer. No pretraining is performed |
| Recursive Language Models (MIT CSAIL) | **Inference-time** recursive self-call over a prompt | Analog of iterated refinement with a stopping rule | Not a representation; nothing is recursively self-prompted here |
| HF `lamm-mit/MetaMaterialsDiscovery` | Research **archive** of agent-built computational labs (Euler–Bernoulli / Timoshenko beam solvers) | Methodological only: state assumptions, run numerical checks, pre-register predictions | **Never cited with an arXiv ID** — its own BibTeX is the placeholder `eprint={xxxx.yyyyy}`. Not an architecture |

**Corpus status is `BLOCKED_CORPUS_AUTH_STALE`.** NotebookLM `auth_status=stale`. **No corpus
citation appears in this pre-registration or in the carrier.** Levin basal-cognition and HaPPY
material were not consulted and are not referenced.

## Acceptance / rejection summary

| Gate | Pass condition | Predicted | If it fails |
|---|---|---|---|
| K1a | separation ≥ 1e-3, control FAILS | PASS | carrier binding is content-blind → reject carrier |
| K1b | exact determinism | PASS | nondeterminism → reject |
| K1c | true shift ≤ 0.5 × median wrong | PASS | roll factor unsound → drop roll factor |
| K1d | enclosure separable, not mass-monotone | PASS | mask factor unsound → drop mask factor |
| K2 | hit ≥ 0.60 and margin ≥ 0.30 over identity | **UNCERTAIN — genuinely unknown** | report negative result; do not add coherence terms to mask it |
| K3 | dead fires, conserved does not | PASS | gate vacuous → reject gate use |
| K4 | incomplete write raises | PASS | DAG not fail-closed → reject migration |

**K2 is the honest unknown.** K1, K3, K4 are predicted to pass on measured evidence. K2 has no
prior measurement — it is the actual experiment. Its result is reported whatever it is.
