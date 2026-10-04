# Blueprint Review — HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3

**Status:** EXTERNAL, CRITICALLY REVIEWED. Adopt the architecture proposal, not the autopsy.
**Document:** `design/zone_a/evidence/HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3.md`
**sha256:** `9f3cd5cd6116e8e2f1c1f77a0c2ae9c2fc03f6c9dbd90bf59b0243af4ce6b7bf`
**Reviewer:** henri-arbiter. **Date:** 2026-10-03.

The blueprint is an **externally authored hypothesis set**. It carries no derivation for its
numbers and no provenance in this repository. It is not a spec. Treat every claim as testable.

---

## 1. Claims the evidence CONTRADICTS

| # | Blueprint claim | This project's measured state | Action |
|---|---|---|---|
| C1 | §1.1 "SpecContract A kill A-K4 ... binding constraint" | **A-K4 PASSED**: 0.000 → 0.833 → 0.083, determinism 1.000 (seal `#84861c41916fb4e5`) | Correct in all downstream docs. The blocked gate is **A-K5 Arm B**. |
| C2 | §1.4 "flat reductionism ... a single linear operator cannot compute multi-step tasks" | Untested as stated. The established defect is a **degenerate training target**: random waves, random labels, salted `hash(text) % 32000`, 35/32000 classes | Record as the blueprint's HYPOTHESIS, not a finding. |
| C3 | §2.2 "training under cross-entropy ... converges to majority-class attractor" | The observed collapse came from an **ungated artifact**, not from a clean linear-head experiment | K-cause test decides this. |
| C4 | §5 Gate G2 `top1_token_unique ≥ 14/16` | The M1 gate **falsified this metric family**: random control 0.592 beat treatment 0.233 | G2 MUST pair with a content-destroyed control and gate on **separation**. |

**CAUTION.** C4 is the highest-risk item. Re-adopting G2 verbatim re-introduces a metric this
repository already refuted. Every gate in the Path A design carries a same-family control.

---

## 2. Claims INTERNALLY INCONSISTENT

| # | Claim | Measurement |
|---|---|---|
| C5 | §2.4 `beta* = 26.10`, `T* = 0.038316` | `1/26.10 = 0.038314` — mismatch at the **5th decimal**. The pair is mutually inconsistent. |
| C6 | §2.0 "`m = diag(m) + A S B^H` beats factorized low-rank by **+0.6083**" | No provenance. Sits **0.0018** from this repo's measured `+0.6065` (gate-power v2, seal `#11dd5dde84db22fb`). **Do not cite 0.6083**; it is not independent evidence. |
| C7 | Stage 1 "incommensurate carriers `omega_k = omega_0 * gamma^k`, `gamma` irrational" | Does **not** survive float. At `D=65536`, near-integer relations recur and reintroduce the harmonic collisions the scheme prevents. |

---

## 3. Technical catch — the similarity kernel DEVIATES

Blueprint Stage 5 scores with `Re(Psi^dag M_k)` — the **real part**.
This repository's validated kernel is `|mean(conj(a) * b)|` — the **complex magnitude** (Sagnac).

**Measured (probe Q3, fixed run):** under a global phase offset of `pi` on the query,
the `Re()` path collapses to **0.000** accuracy. A wave system has **no absolute phase**.
A global phase offset carries no information, so any kernel that a global phase can invert
introduces an unvalidated failure mode into the one component whose job is discrimination.

**Decision:** the snap uses the **magnitude kernel** unless a run justifies `Re()`. Deviation
must be argued with numbers, not imported silently.

---

## 4. Claims the evidence SUPPORTS — adopt these

- **Strongly typed manifolds (`K <= 512`) over 32,000 unconstrained tokens.** The strongest idea
  in the document. It sidesteps the degenerate-vocabulary problem at its root. Make it the core.
- **Hopfield energy readout over flat linear projection.** A defensible inductive-bias upgrade.
  Directly testable against the already-validated gate.
- **G1 no `[D,D]` allocation, `<2 GiB`.** Already proven: `probe_full_D_no_DxD_state` PASS.
  Codebook cost: `K=512`, `D=65536`, complex64 = **256 MiB**. Fits host and device.
- **Reuse `ContinuousHopfieldCleanup`.** Already measured at p50 252.26 us. Extend, do not rebuild.
- **Stages 1–4.** Ingress, wave core, Sagnac veto, and Zone C **already exist and are verified**
  this session. The blueprint's §1.2 and §1.3 restatements match our measurements. No work needed.

---

## 5. Scope discipline

The blueprint names exactly **one** Immediate Feasible Action (§2.4): resolve the
**Egress Transduction Interface**. Stages 1–4 are built. Stage 6 is downstream of a passing
egress gate.

**Rejected reading:** build all six stages. That would place unverified work in the repository.
`K <= 512` typed manifold + Hopfield readout + validated control is this session's scope.

---

## 6. Latency

Gate G5 (`50 us`) stays **deprioritized** per standing operator instruction. The blueprint
does not re-activate it.
