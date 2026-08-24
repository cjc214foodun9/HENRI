# System-1 v0.6.4 — Continuous-Learning Carrier Premise Audit (NO-GO as specified)

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding**
**Upload:** `v0.6.3_Post-Mortem___Continuous_Learning_Blueprint.md`
sha `db651ce1…` (977 B) — 8-line degeneracy diagram (no C1–C8 enumerated in
the artifact; the C1–C8 in the request text are unstated).

## Upload dispositions

| Claim | Disposition | Evidence |
|---|---|---|
| v0.6.3 degeneracy understood (H/logK≈0.9999) | **VERIFIED** | Matches smoke063 measurement exactly |
| CUDA cancellation was correct | **VERIFIED** | Reference 3 gate; margin=0 audit |
| "Real continuous learning requires real error feedback" | **CONDITIONAL** | True in general; false as EDMD signal — see corpus #21 |
| v0.6.4: verifier rejections V(p_k)=0 drive O(r²·D) R-EDMD fast-weight adaptation during episode unrolls | **FALSIFIED / BLOCKED_MISSING_PREMISE** | (1) corpus #21: scalar labels cannot drive EDMD — requires state transition pairs (x,y)=(z_{l-1},z_l); (2) live v0.5.5 carrier has **0** EDMD/wave/engram hits in 282 lines; (3) **no episode unroll substrate** — candidate scan is a static set per task, no temporal dynamics, no reset semantics, no branch-local trajectories; (4) D=65,536 does not exist — live is [1,16,384]; a 65,536-D bank is a third tensor family (v060 bridge already documents this) |
| O(r²·D) complexity | **CONDITIONAL** | Formula A_{t+1}=λA_t+ψψᵀ, λ∈[0.95,0.99] is corpus-approved — but no live substrate consumes it in System-1 |

## Reference 3 — C1–C8 pre-registration (not construction)

| # | Contract | Status |
|---|---|---|
| C1 | Real live-dimensional transition provenance — explicit (x_t, x_{t+1}) pairs, control/context, target construction, temporal ordering | BLOCKED — cannot construct: no dynamical state in the DSL loop |
| C2 | Verifier/outcome partition separation (disjoint fixtures) | CARRIED from v0.5.5 protocol (satisfiable when substrate exists) |
| C3 | η=0 byte-identical baseline | CARRIED (satisfiable) |
| C4 | Deterministic reset + branch-local isolation | BLOCKED — no episode/branch structure exists |
| C5 | Bounded spectral radius/condition number | BOUNDED_IMPLEMENTABLE on a real substrate; no substrate now |
| C6 | Rollback on preservation/cost violation | CARRIED (satisfiable) |
| C7 | Non-vacuous engagement + measurable first-pass-rank effect | CARRIED (satisfiable) |
| C8 | Exact outcome/per-family preservation, matched runtime/VRAM/calls | CARRIED (satisfiable) |

## Corpus consult #21 (INFERRED, 13 sources, primary bank)

- EDMD is algebraic regression of a Koopman operator: needs **state transition pairs**.
  Binary labels contain no spatial/phase degrees of freedom → cannot compute the
  cross-covariance terms. Their proper role: trajectory-level REINFORCE or
  thermodynamic gating thresholds — not operator regression.
- Approved update: A_{t+1} = λA_t + ψψᵀ, λ∈[0.95,0.99]; low-rank factors
  V,W ∈ ℂ^{D×r}; N ≥ r independent triples needed for identifiability.
- Capacity walls: M_max=280 superposed engrams @ D=65,536 (Sagnac veto spikes
  beyond); N_max≈9,044 zero-entropy engrams; Δ_Sagnac > 0.35 ⇒ Langevin loop.
- Learning must be **gated on demonstrated failure regimes** (T4 gate:
  explore iff loss_ema > accuracy_floor). No unconditional attachment.

## Decision

- **NO-GO** on constructing `v064_continuous_learning_carrier.py` now.
- The System-1 bounded-DSL loop has **no dynamical substrate** for EDMD:
  static candidate sets, scalar rejection labels, no episode state, no
  transition pairs, live dim [1,16,384].
- The audit and C1–C8 pre-registration are the deliverable. Construction is
  authorized only if a real dynamical environment exists (VLA staged roadmap
  stage 1+ substrate, per audit decision (3) of the VLA cycle) — never as a
  wrapper over the current carrier.
- A verifier-conditioned fast-weight carrier, even if built later, is bounded
  program-synthesis adaptation — **NOT VLA/continuous-environment learning**
  (Reference 3 point 8; VLA gate remains 0/12).

## Next falsification

Any claim that binary verifier rejections can drive R-EDMD operator learning
in the current 13-family loop. Falsified by: corpus #21 (scalar-label
inadequacy), 0 EDMD substrate in live carrier, no episode structure.
