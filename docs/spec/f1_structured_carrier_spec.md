# F1 Structured Carrier Specification (SpecContract A)

**Spec ID:** SPEC-2026-08-28-F1-CARRIER
**Author holon:** /henri-research (Exploration & Spec)
**Consumer holon:** /henri-architecture (Implementation Engine)
**Source directive:** HENRI-DIR-2026-08-M2-POSTMORTEM-F1-STRATEGY (inbox SHA-256 `1d2de7ccd2c165dfeb6a7f4e7b385a72e7006a5ff13642600ca179c99292b40a`, 16,134 bytes, OBSERVED)
**Status:** STAGED — pre-registration sealed before any active-code change
**Governance:** M2 closure seal `#b80bbd8f` verified at ledger idx 959 (`M2SUCC_V3_VERDICT`, carrier_sha `3702af9`, OBSERVED)

---

## 1. Academic foundation

### 1.1 Why M2 failed (verified, not assumed)

M2 verdict reproduced exactly from live telemetry (reducer `m2_coherence_reduce.py` against `henri_r2_next/m2_gauntlet_v3`, 18 cells, OBSERVED):

| Quantity | Value | Gate | Status |
|---|---|---|---|
| Main telemetry rows | 951 | — | OBSERVED |
| Engaged rows | 933 (rate 0.9811) | ≥ 0.95 | PASS |
| Cohort Δ̄(k=1) | 0.983192 | ≤ 0.15 | FAIL |
| Cohort Δ̄(k=4) | 1.006124 (degraded peak) | ≤ 0.15 | FAIL |
| Verdict | M2_HORIZON_COHERENCE_FALSIFIED | — | SEALED #b80bbd8f |

Mechanism of failure: the rank-128 factorized Koopman operator projects into a
subspace spanning < 0.2% of the active tangent space. Unrolled predictions
`K_a^k Ψ_t` fall into the orthogonal complement: at D = 65,536 the expected
|cos| of independent vectors is `sqrt(π/4D) ≈ 0.00346`, so Δ ≈ 0.9965 — exactly
the observed floor. This is measure concentration on S^(D-1), not operator
tuning error.

### 1.2 The F1 hypothesis

Action must act as a **differentiable Lie displacement generator** that rotates
the state along a structured geodesic instead of pushing it into the orthogonal
complement:

```
Ψ_{m,t+1} = R(θ_{m,t}) Ψ_{m,t},   R(θ_m) = exp(Σ_a θ_{m,a} M_a) ∈ SO(8)
```

per block `m ∈ [1, 8192]`, with `θ_m ∈ R^8` the goal-conditioned displacement
coefficients and `M_a` the **adjoint su(3) generators**.

**Falsifiable claim:** a block-wise, norm-preserving, goal-conditioned Lie
displacement preserves phase coherence across k ≥ 4 unroll steps where the
rank-128 EDMD path disperses to Δ ≈ 1.0.

### 1.3 Adjoint su(3) basis (VERIFIED numerically, OBSERVED 2026-08-28)

From the live Gell-Mann basis (Tr(λaλb) = 2δab, `chromodynamic_grounding.py`) and
structure constants `[λa,λb] = 2i·f_abc·λc`:

```
(M_a)_{cb} = -f_abc          (adjoint generator, real 8×8)
R(θ) = exp(θ^a M_a) ∈ SO(8)
```

Numerical probe (CPU torch fp64, seed 20260828):

| Check | Result | Gate |
|---|---|---|
| C1 skew: M_a + M_aᵀ | 0.000e+00 | exact |
| C2 orth: RᵀR − I | 7.378e-15 | ≤ 1e-10 |
| C3 Ad-equivalence sign | err(+M) 6.240e-15 vs err(−M) 3.683e+00 → **sign = +M** | anchor |
| C4 Hadamard ⊙ of skew | symmetric 0.0; exp(⊙) orth err 9.1e10 | **FALSIFIES doc F1.2** |
| C5 Procrustes → Log → Π_adj recovery | fit 1.766e-15, rollout 5.301e-15 | exact |

**Boundary:** the generator family `{exp(θᵃMₐ)}` is `Ad(SU(3)) ≅ SU(3)/Z₃`, a
proper 8-parameter subgroup of SO(8). It is NOT full O(8). Do not claim general
rotation coverage (O(8) gauge claim remains FALSIFIED per G0/G1 2026-08-25).

### 1.4 Goal-conditioning — corrected coupling (F1.2 amendment)

The directive's Phase F1.2 formula `D_a ← D_a ⊙ proj_su3(W_task)` is
**FALSIFIED**: the Hadamard product of two skew matrices is symmetric
(C4: sym_err = 0.0), so `exp` of the product is not orthogonal and diverges.

Replacement (additive in the Lie algebra, preserves closure):

```
θ_{m,t} = θ_base,m + λ · Π_adj(W_task)_m        (per-block, λ ≥ 0)
Ψ_{m,t+1} = exp(θ_{m,t}^a M_a) · Ψ_{m,t}
```

`Π_adj(W_task)_m` = projection of the compiled task functor onto the adjoint
algebra per block: `θ_task,m = Π_adj(Log(R_task,m))` via the verified C5
recovery (Procrustes fit → matrix log → adjoint projection).

## 2. Tensor contracts

| Tensor | Shape | Dtype | Dynamic axes | Notes |
|---|---|---|---|---|
| `state_wave` | `[num_blocks, 8]` | float32 | num_blocks | canonical planner boundary; real |
| `theta` (displacement) | `[num_blocks, 8]` | float32 | num_blocks | 8 adjoint coefficients per block |
| `R_block` | `[num_blocks, 8, 8]` | float32 | num_blocks | exp(θᵃMₐ); must be orthogonal |
| `w_task_adj` | `[num_blocks, 8]` | float32 | num_blocks | projected functor; default zeros |
| `action_wave` | `[num_blocks, 8]` | float32 | num_blocks | existing boundary, unchanged |

- No `[D,D]` or `[8·num_blocks, 8·num_blocks]` dense operators. Per-block 8×8
  only: 8192 × 64 × 4 B = 2 MiB per rotation tensor.
- Accumulation dtype: fp32. The 8×8 kernel is too small to need bf16; do NOT
  introduce mixed precision without a new pre-registration.
- Kernel implementation emits an implementation marker:
  `TRITON` | `TORCH_REF` — the CUDA gate requires `TRITON`; a CPU or torch
  fallback result is labeled `BLOCKED_VACUOUS_CPU_PATH`, never PASS.

### 2.1 Triton kernel contract

- Per-block 8×8 real matrix exponential of a skew matrix, unrolled (no list
  comprehensions, no `tl.static_range` list building — Phase 8.18 D24 rule).
- Scaling-and-squaring Taylor: scale s = ceil(log2(‖θᵃMₐ‖)) so ‖S/2^s‖ ≤ 0.5,
  Taylor degree 6, square s times. Row stride MUST be multiplied in
  (Phase 8.18 D27 rule: `base = offs * 64` for [N,8,8] row-major).
- Clamp radicands/norms: `tl.maximum(x, 0.0)` (Phase 8.18 D26 rule).
- Complex dtype promotion trap: this kernel is real-only; no `1j * tensor`
  promotion hazard (Phase 8.18 D23 rule does not apply to the real path).

## 3. Mathematical invariants (falsifiable)

1. Per-block row norm preserved: `|‖Ψ'_m‖₂ − ‖Ψ_m‖₂| ≤ 1e-4` across the 8-step
   unroll (orthogonal displacement).
2. `R_mᵀ R_m − I₈` Frobenius ≤ 1e-4 per block, every step.
3. Normalized Sagnac delta stays in [0, 2]: `1 − cos(Ψ̂, Ψ)`, clamped.
4. Displacement family = Ad(SU(3)); θ = 0 ⇒ R = I (identity arm byte-identical
   to baseline path — the default-OFF differential).
5. λ = 0 ⇒ carrier output byte-identical to baseline transition output
   (mechanism inertness proof).
6. Sign convention fixed: `(M_a)_{cb} = −f_abc`, `R = exp(+θᵃMₐ)`.
7. No trainable parameters in F1.1–F1.3. θ is compiled closed-form (C5) or
   demo-pair-derived. SGLD creep is a LATER carrier, pre-registered separately.

## 4. Loss formulations

None. F1.1–F1.3 is a zero-trainable representation carrier. Telemetry-only
signals: per-step Δ_Sagnac by horizon (reuse `henri_m2_coherence.py`),
per-block θ norm, λ, R orth error, implementation marker.

## 5. Baseline references

| Reference | ID / Path | Core insight |
|---|---|---|
| Phase 8.18 SU(3) transducer | SEALED ACCEPT 2026-08-17, G1 5.5e-7, G2 227.4, G3 27.8µs | Triton 3×3 complex kernel; D23–D27 trap catalog |
| M2 standing order | HENRI-ORD-2026-08-M2-COHERENCE-REDUCTION | pending-buffer horizon telemetry; reducer contract |
| M2 verdict | seal #b80bbd8f, idx 959 | unconditioned EDMD dispersion baseline |
| Yang–Mills↔qFHRR bridge | `yang-mills-qfhrr-bridge.md` | Sagnac = U(1) Wilson action; SU(3) NOT in Cl(3,0)/Cl(1,3) |
| Hopfield egress (M3) | `hopfield_cleanup.py` | flat-softmax degeneracy at Δ≈1 (M3 gated) |
| G0/G1 gauge audit | sealed 2026-08-25 | O(8) gauge FALSIFIED; row-norm invariants INFO_COLLAPSE; sanctioned egress = per-block cosine + joint frame transform |

## 6. Failure modes

1. Generator family cannot carry the dynamics (Ad(SU(3)) too small for the
   observed transitions) → FALSIFIED at C5-fit rollout or K3 gate.
2. θ fit overfits the calibration episodes and does not transfer → K4 shuffle
   control must show no gain.
3. Per-block SO(8) displacement destroys cross-block coupling (the M2
   transition is coupled low-rank) → measure cross-block Jacobian; if the
   coupled signal vanishes, the carrier needs a coupled-θ variant (bounded
   amendment, NOT silent).
4. Triton 8×8 exp precision loss in fp32 → gate vs torch fp64 reference at
   ≤ 1e-4 relative; failure = kernel defect, not mechanism verdict.

## 7. Cheapest kill experiments (pre-registered)

- **K1 (norm/orth):** 8-step unroll on frozen waves; orth error ≤ 1e-4 and
  per-block norm drift ≤ 1e-4. Failure = kernel defect (BLOCKED_INFRA, not
  mechanism verdict).
- **K2 (identity arm):** θ = 0 ⇒ outputs byte-identical to baseline; proves
  default-OFF and isolates the displacement as the only changed channel.
- **K3 (fit transfer):** fit θ per env on calibration episodes (C5), evaluate
  on disjoint eval episodes. Gate: one-step Δ̄ < 0.5 (vs M2 baseline 0.983).
  Failure = FALSIFIED_NO_TRANSFER (generator family cannot carry dynamics).
- **K4 (shuffle control):** per-block θ assignment permuted across blocks.
  Gate: no coherence gain vs unpermuted (Δ̄_shuffled − Δ̄_fit ≥ 0.15).
  Failure of the gate to separate = the fit is not causal → FALSIFIED.
- **K5 (engagement):** ≥ 95% of main telemetry rows carry finite horizon
  deltas for k ∈ {1..8} (same standard as M2 STEP-2).

## 8. F1.3 gauntlet gates (pre-registered acceptance)

Re-run the 18-cell M2 coherence matrix under `HENRI_F1_CARRIER=1`:

| Gate | Criterion | Status |
|---|---|---|
| G1 liveness | 18/18 cells RC=0 | required |
| G2 engagement | ≥ 0.95 engaged rows, all k ∈ {1..8} nonzero | required |
| G3 primary coherence | cohort Δ̄(k) ≤ 0.35 for k ∈ {1..4} | ACCEPT criterion |
| G4 secondary diagnostic | cohort Δ̄(k) ≤ 0.35 for k ∈ {5..8} | diagnostic only |
| G5 baseline beat | Δ̄(1) < 0.5 (vs M2 0.983) | required |
| G6 identity differential | flag-absent path byte-identical | required |
| G7 kernel truth | implementation marker = TRITON on all cells | required |

NOTE: F1.3's 0.35 / k∈{1..4} gate is a RELAXATION of M2's 0.15 / k∈{1..8}
pre-registered gate. This is the directive's stated F1.3 gate and is adopted
verbatim; the change is disclosed here, not silent.

DIAGNOSTIC (amendment 2026-08-28, additive, no gate change): telemetry adds
`f1_drift_slope` = least-squares slope of cohort Δ̄(k) over k ∈ {1..8}.
A non-positive slope is the "phase drift eliminated" reading; a positive slope
is residual drift. It interprets the verdict only; it never passes G3.

Verdicts: `F1_CARRIER_VERIFIED` | `F1_CARRIER_FALSIFIED` | `BLOCKED_INFRA` |
`FALSIFIED_NO_TRANSFER` | `FALSIFIED_NO_ENGAGEMENT` (M2 taxonomy).

## 9. Precision and execution constraints

- All tensor ops fp32; accumulation fp32; no bf16 without amendment.
- CUDA-only verification. Local CPU tests verify shapes/differential only and
  never count as the CUDA gate (accelerator-gate-validity rule).
- Triton API probe BEFORE kernel commit (Phase 8.18 D25: `tl.math.atan2`
  absent; probe remote surface first).
- CPU-generator → CUDA placement: generate on CPU, `.to(device)` explicitly
  (Phase 8.18 lesson 5).
- `torch.linalg.matrix_exp` exists; `torch.linalg.matrix_log` does NOT on
  torch 2.12+cu130 — use eigendecomposition log for the reference (Phase 8.18
  lesson 1).

## 10. Interface (HarnessContract input, immutable for F1)

```python
# f1_carrier.py (default-OFF; HENRI_F1_CARRIER=1)
class F1LieDisplacementCarrier:
    def __init__(self, num_blocks: int, device: torch.device): ...
    def compile_theta(self, w_task_adj: torch.Tensor | None,
                      theta_base: torch.Tensor | None = None,
                      lam: float = 0.0) -> torch.Tensor: ...   # [num_blocks, 8]
    def step(self, state_wave: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Psi' = exp(theta^a M_a) Psi per block. Returns [num_blocks, 8] real."""
    def fit_adjoint(self, traj: torch.Tensor) -> torch.Tensor:
        """Closed-form per-block theta from observed transitions (C5)."""
```

Entrypoint for the gauntlet: `production_arc_run.py --envs <n> --steps 60` with
`HENRI_F1_CARRIER=1`; telemetry keys `f1_engaged`, `f1_theta_norm`,
`f1_orth_err`, `f1_impl`.

## 11. Escalation

- Harness failures (kernel, placement, dtype): Contract C → /henri-architecture
  targeted AST-diff fix, iteration ≤ 2, then Sol gate.
- FALSIFIED_NO_TRANSFER or FALSIFIED at G3: Contract C → /henri-research
  re-derives invariants (coupled-θ variant or different generator family).
  Do NOT tune λ or θ scales to pass a failed gate.
