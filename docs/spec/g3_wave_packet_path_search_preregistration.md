# Carrier G3 — Wave-Packet Path Search: Pre-Registration (Diagnostic Sidecar)

**Directive:** user approval (2026-09-01, "Yes — G3 diagnostic sidecar with audited corrections") + `holographic search.pdf` (`HENRI-AUDIT-2026-09-V3-QUANTUM-WAVE-SEARCH`, 190,418 B, SHA `76c28f6b69f1…`, ledger @1,162).

## Mechanism

`WavePacketPathSearch`: superpose all action paths as phase-rotated wavefronts
(`Ψ_next = Ψ_curr ⊙ exp(jθ_a)`), apply Sagnac homodyne clearance against Zone C
invariant axioms (destructive interference annihilates invalid trajectories),
keep top-k coherent modes, iterate to horizon H. Replaces classical sequential
MCTS expansion with a single batched tensor pass per step.

## Audited corrections vs the PDF (load-bearing)

| PDF claim | Disposition |
|---|---|
| `action_generators = nn.Parameter(torch.randn(A, D, complex64) * 0.01)` | **REJECTED** — violates zero-trainable invariant. Frozen deterministic seeded generators (SEED 20260926), plain buffers, zero `Parameter`s. |
| `sagnac_energy = sin(phase_mismatch·π/4)²` | **REJECTED** — not the HENRI Sagnac delta. Use HENRI normalized Sagnac: `Δ = 1 − Re(⟨a,b⟩)/(‖a‖‖b‖) ∈ [0,2]` (Cauchy–Schwarz). Threshold default 0.05 (PDF's 0.035 was calibrated to its own formula). |
| Complex flat `[D]` wave family | **THIRD family** (invariant) — diagnostic sidecar ONLY, one-way norm-preserving complexification adapter from canonical real `[8192,8]`, NO policy influence. |
| "replace discrete MCTS" | **NOT in this carrier.** Sidecar diagnostic only; planner wiring = separate carrier with its own prereg + approval. |
| Codebase audit claims (sagnac_mcts_planner = live planner) | **STALE** — live action path is `EFEPlanner.select_action`; MCTS is not the live planner. |

## Data path

`real [8192,8] wave → complexify (imag=0) → unit normalize → [B, D=65536] complex → propagate_superposed_paths(Ψ, priors, axioms) → (best_wavefront, coherence, clearance)`.
Default path byte-identical (flag `HENRI_G3_WAVE_PACKET=1` required; module never imported by production code).

## Bounds

- Dim: 65,536 remote CUDA; 4,096 local CPU. A=7, H=8, top-k ≤ 64, seed 20260926.
- Threshold 0.05, cavity temperature 0.05.

## Gates (pre-registered)

- **C1 default-OFF differential:** without `HENRI_G3_WAVE_PACKET=1`, construction raises `require_flag` error; production runner source does not import the module.
- **C2 frozen determinism:** two instances, same seed → byte-identical generators; zero `nn.Parameter` in module state.
- **C3 norm preservation:** every expanded path unit-norm (max err ≤ 1e-5) after each step.
- **C4 veto selectivity:** axiom aligned to action a ⇒ path-a survival ≥ 0.95; orthogonal actions pruned ≥ 0.95 (HENRI Sagnac delta).
- **C5 top-k bounded:** active paths ≤ 64; best path = argmax coherence.
- **C6 adapter:** one-way complexification preserves norm (err ≤ 1e-5) and is injective.
- **C7 no-policy influence:** static + differential proof that production action selection is unchanged.
- **C8 latency (diagnostic, CUDA):** per-step ms and total expanded paths reported vs sequential MCTS baseline; not a kill gate for a diagnostic sidecar.

## Kill criteria

Any C1–C7 failure → `G3_SIDECAR_FALSIFIED`. All pass → `G3_SIDECAR_VERIFIED` (diagnostic only; NO capability or benchmark claim; planner wiring remains a separate approval-gated carrier).

## Cheapest kill experiment

C4 veto selectivity on 2 actions × 2 axioms (aligned vs orthogonal) at D=4096 CPU. Fails ⇒ no CUDA run needed.
