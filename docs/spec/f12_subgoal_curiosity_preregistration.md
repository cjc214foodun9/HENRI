# Carrier F12 Pre-Registration — Sub-Goal Phase Hashing & Intrinsic Curiosity Ingress

**Directive:** HENRI-DIR-2026-08-F11-POSTMORTEM-SUBGOAL-INGRESS
**Directive SHA-256:** `f19107fdcabc90a1b109d879328c4a77a65ed432599c2ed91135331818888a4d` (18,149 B)
**Carrier:** F12 — Intrinsic Curiosity Active Inference Engine
**Branch:** `carrier/f12-subgoal-curiosity`
**Engine:** `HENRI V2/experiments/verification/arc_f12_curiosity_engine.py`
**Receipt schema:** `f12-curiosity.v1`
**Flag (default-OFF, fail-closed):** `HENRI_F12_CURIOSITY=1`

## Status

F11 verdict RATIFIED (`F11_GATE_G2_FAILED`, seal `#95c17ee1`). Formal GO granted for Carrier F12.

## Mechanism (per directive §3)

- **Tier 1 — Single-Pass Wave-JEPA Dynamic Rollout:** for each candidate action `a ∈ {1..8}`,
  `Ψ̂_{t+1}(a) = exp(D_a) · Ψ_t`, `D_a` skew-symmetric (seeded, zero-trainable) so `exp(D_a)` is
  orthogonal and norm-preserving on `S^{D-1}`.
- **Tier 2 — Intrinsic Free-Energy Action Selection:**
  `a* = argmax_a [ λ_cur · (1 − |⟨Ψ̂_{t+1}(a), Ψ_axiom⟩|) + λ_nov · 1/√(N(h_{t+1}(a))) ]`
- **Tier 3 — Online Hebbian Trace Update with Dense Intrinsic Valence:**
  `Δν_t = r_ext(t) + λ_cur·r_surprise(t) + λ_nov·r_novelty(t) > 0 ⇒ M_a ← Normalize_L2(M_a + η_fast·Δν_t·Ψ_t)`
- **Tier 4 — Fast State Hash Frontier Cache:** `N(h) ← N(h) + 1` for `h = Hash(Ψ_{t+1})`.

Intrinsic channels:

```
r_surprise(t) = 1 − |⟨exp(D_{a_t})·Ψ_t, Ψ_{t+1}⟩|      (cosine-normalized, [0,1])
r_novelty(t)  = 1 / √(N(Hash(Ψ_{t+1})) + 1)              (count BEFORE increment, D7)
r_ext(t)      = Δlevels_completed − 0.5·reset             (D2, carried from F11)
```

## Gates (directive §4)

| Gate | Requirement | Kill |
|---|---|---|
| G1 | 1,800 steps (12 envs × 150), mean latency ≤ 5.0 ms/step | K1 Inference Latency Regression |
| G2 | ≥ 1 of 12 live envs solved end-to-end (score > 0.0%) | K2 Zero Task-Solving Emergence |
| G3 | cohort Hebbian creep updates `creeps ≥ 100` | K3 Intrinsic Valence Dormancy |
| G4 | mean Sagnac ≤ 0.050 (F10/F11-comparable convention, D9) | K4 Coherence Degradation under Curiosity |

Verdict map: `F12_CURIOSITY_LOOP_VERIFIED` (all pass) | `F12_GATE_G2_FAILED` / `F12_GATE_G3_FAILED` /
`F12_GATE_G4_FAILED` (first failing of G2→G3→G4) | `F12_LIVE_ENGINE_BLOCKED` (G1 or infra).

## Bounded execution command (directive §5)

```bash
python HENRI\ V2/experiments/verification/arc_f12_curiosity_engine.py \
    --device cuda --steps-per-env 150 --seed 20260910 \
    --lambda-curiosity 1.0 --lambda-novelty 0.5 --eta-fast 0.05 \
    --out-dir /tmp/henri_f12_curiosity/ \
    --receipt-out /tmp/henri_f12_curiosity/f12_gates_receipt.json
```

## Pre-flight kill contract (K2 rule)

Synthetic in-sample test before any live launch: the dense intrinsic valence channel alone
(r_ext = 0, surprise + novelty) must produce `Δν > 0` and `creeps > 0` on a constructed
novel-state stream (contract C7). Failure to engage on synthetic data ⇒ do not launch live.

## Deviations (disclosed, D-series)

- **D1:** `Ψ_axiom` = episodic first-frame wave (goal). No Zone C DSN is provisioned in the
  directive's command; boundary axioms are unavailable in the compact engine context.
- **D2:** `r_ext` mapped from the live Arcade API (no `reward`/`score` field):
  `Δlevels_completed − 0.5·reset` (F11 D2 carried).
- **D3:** ledger offset — directive cites F11 verdict at record 1,075; live verdict is #1074
  and this directive's ingest event is #1075. Target record 1,076+ remains valid.
- **D4:** compact verification substrate D=64 (num_blocks=8, ratified F10/F11 scale);
  `M ∈ R^{8×64}`, `D_a ∈ R^{8×64×64}` precomputed once at init.
- **D5:** surprise is cosine-normalized: per-block unit-norm waves have flat norm √8;
  `r_surprise = 1 − |cos| ∈ [0,1]` (equivalent to the directive's `1 − |⟨·,·⟩|` for unit vectors).
- **D6:** `h = crc32(packed sign bits of flat Ψ) ∈ Z_{2^32}`; count table is in-memory
  (L2-resident per directive §3.1.2).
- **D7:** novelty uses `N(h)` BEFORE increment; Tier-4 visit increments AFTER reward computation.
- **D8:** Tier-3 prototypes `M` are diagnostic in this carrier — the directive's Tier-2 formula
  does not consume `M`. Prototype cosines are recorded in telemetry; a selection consumer
  requires a future carrier. (G3 remains an engagement gate for the plasticity machinery itself.)
- **D9:** G4 coherence convention = F10/F11-comparable mean `sagnac_delta(roll[0,0], goal)`
  with `goal` = episodic first frame.

## Determinism

Module-level seeded initialization (generators `seed`, `seed+1`, `seed+10`); zero-trainable
buffers only; frontier hash deterministic (numpy packbits + crc32). A sealed carrier must be
reproducible.
