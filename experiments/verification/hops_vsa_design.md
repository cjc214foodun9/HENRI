# Class 4.5 HOPS-VLA Module — Bounded Design, Contradiction Matrix & Pre-Registration

Doc ID: HENRI-CLASS45-HOPSVLA-MODULE-2026-08-21
Authoritative spec: `/g/My Drive/HENRI_Inbox/HENRI_Ground-Up_VSA_Holographic_Model_Specification.md`
(SHA `c304d30d3d363572a2d02f89ff352b7e4c7bc6ec821beb2185b0f7e988f768b1`, 87 lines)
User instruction (2026-08-21, mid-turn): add the HOPS-VSA module to the development
process while MAINTAINING the Path B2 pre-training plan and its documents.

## 1. Scope (included / excluded)

INCLUDED (this phase, default-OFF Python module `hops_vsa_core.py`):
- A. Ingress invariant-subspace decoupling: P_null = I - V V† (V = orthonormal
  skeleton basis from common AST node types) — carrier-overlap removal.
- B. Diagonal complex Clifford rotor (exp(i·b) per phasor) with exact isometry
  (Gram < 1e-6) — the spec's E_Gram bound.
- C. Dual-channel Sagnac veto over the skeleton-free channel (Δ > 0.35 veto).
- D. Candidate scoring over the skeleton-free channel (margin deformation).
- E. Contract tests (9) + runner wiring `--hops-vsa-rank` default OFF.

EXCLUDED (deferred, separate gated phases per the spec's own plan):
- Fused C++/CUDA kernel `hops_vla_cuda_core.cu` (Phase 1 of the spec; requires
  remote compile toolchain + Metric M4 latency gate; GPU-verify after B2 frees
  the 5090).
- Zone C schema DDL + `seed_zone_c_timescaledb.py` + >1M engram seeding
  (Phase 3; NO live Zone C writes in this phase; supplied seeding module from
  the prior packet remains audited-but-not-executed).
- Full AAII v4.1 campaign (Phase 4).

## 2. Contradiction matrix (spec vs live code vs sealed record)

| Spec mechanism | Live-code anchor | Status |
|---|---|---|
| Carrier overlap (E[cos]≈0.59) | `qfhrr_ast_discriminative_kernel.py` carrier_subtract (default False) | EXISTS; module generalizes to P_null projector |
| Complex Clifford rotors | `product_clifford_product_kernel.py` ProductCliffordAlgebra3D | EXISTS; module adds diagonal rotor + Cholesky retraction |
| Sagnac homodyne veto | `henri_vision_encoder.py:165` compute_sagnac_similarity | EXISTS; module reuses over P_null channel |
| E_Gram 1.19–1.48 | Sealed Path A verdict (a_orth_error) | CONFIRMED FALSIFIED; module enforces < 1e-6 |
| Gate A margin −0.0838 | Sealed Path B verdict | CONFIRMED; module does NOT claim margin; Gate A/B of Path B2 measure it |
| hops_vla_cuda_core.cu | PHANTOM (no file) | FORWARD step, excluded |
| Zone C 1M engrams | NO live DB writes authorized | EXCLUDED |

## 3. Representation boundary

- Continuous float32 waves `[D]` (interleaved cos/sin), D=65,536; uint8 ring
  input → `RepresentationBoundaryError`. No dense `[D,D]` (banned, 34 GiB);
  projector is `V` `[D, k]` with k ≤ 8 (≈ 4 MB), rotor is diagonal.
- P_null = I − V V†: orthogonal projector (V orthonormal via Cholesky).

## 4. Data path & zero-pretraining boundary

- Skeleton basis: deterministic hash of common AST node types (Module,
  FunctionDef, Return, Name, arg, Call). NO dataset content enters the basis.
- Scorer input: candidate AST source strings → codec/phasor waves (same
  encoder family as the runner; no HumanEval/ARC/eval-cache content).

## 5. Resource limits

- Module params: rotor bivector `[D/2]` (~0.26 MB); projector V `[D, 8]`
  (~4 MB). Total well under 0.1 GB. No optimizer state in inference.

## 6. Expected benefit (falsifiable) + kill gates

| Gate | Criterion | Kill |
|---|---|---|
| G1 (isometry) | rotor Gram < 1e-6 | > 1e-4 ⇒ FALSIFIED |
| G2 (carrier removal) | skeleton-projected residual cos < 0.05 with V | ≥ 0.05 ⇒ FALSIFIED |
| G3 (channel separation) | same-skeleton/different-body: null-channel cos < raw cos | no decrease ⇒ FALSIFIED |
| G4 (veto) | Δ > 0.35 veto fires; Δ=0 no veto | both wrong ⇒ FALSIFIED |
| G5 (external, AFTER B2 Gate A) | paired HumanEval with --hops-vsa-rank improves external passes vs control | no gain ⇒ FALSIFIED_NO_EXTERNAL_GAIN |

G1–G4 are local/CPU contract gates. G5 is the ONLY external-outcome gate and
runs after B2 training completes (GPU-exclusive rule) with its own
pre-registration.

## 7. Relation to Path B2 (user mandate)

- Path B2 plan UNCHANGED: training in flight (remote PID 153778, ckpt contract
  `henri.path-b2-codec.v1`), Gate A/B binding. HOPS-VSA is ADDITIVE: default-OFF
  module + runner flag; no B2 file modified by this phase.
- Naming: the spec calls the architecture HOPS-VLA; the module is
  `hops_vsa_core.py` (Python reference core). The CUDA kernel phase retains the
  spec's `hops_vla_cuda_core.cu` name.

## 8. Execution order

1. Design packet (this file) → 2. module + contracts (RED→GREEN local CPU) →
3. runner flag wiring (default OFF) → 4. commit + push (feature branch) →
5. remote focused contracts after B2 training completes → 6. G1–G4 CUDA
verification → 7. G5 paired external gate (pre-registered, after B2 Gate A).
No main changes; no live Zone C writes.
