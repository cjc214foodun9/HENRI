# Carrier F13 — Hierarchical Wave-Optic Goal Steering & Macro-Action Synthesis: Pre-Registration

**Document Identifier:** HENRI-DIR-2026-08-F12-POSTMORTEM-HIERARCHICAL-STEERING (Carrier F13)
**Directive SHA-256:** `d02eca2cd414bdce52ec5fedad3275214b21414ef273cf97c01e5de9db6b3a2f` (18,857 B, 225 lines, full read)
**F12 Verdict Ratified:** `F12_GATES_VERDICT #5145429145…` = `F12_GATE_G2_FAILED` (Ledger 1,078; ingest of this directive @ 1,079)
**Branch:** `carrier/f13-hierarchical-steering` (from F12 HEAD `5a5000c`)
**Status:** PRE-REGISTRATION (sealed before implementation)

---

## 1. Post-Mortem Ratification

F12 sealed `F12_GATE_G2_FAILED`:
- G1 ✅ 1.67 ms/step (1,800 steps); G2 ❌ 0 solved; G3 ✅ creeps 1,642 (ΣΔν +1174.6); G4 ✅ Sagnac 0.0253.
- Pathology isolated: "Noisy TV" epistemic trap — 91 distinct state hashes, 158 resets, surprise 0.570. Intrinsic curiosity provably engages plasticity but does not convert to task resolution.

F13 replaces undirected curiosity with **directional sub-goal steering**: geodesic waypoint interpolation toward a static goal wave, 8-step macro-action beam search maximizing waypoint phase alignment under a Sagnac validity penalty, and goal-convergence valence driving the Hebbian memory.

## 2. Gates (pre-registered, from directive §4)

| Gate | Criterion | Kill |
|---|---|---|
| G1 | 1,800 steps (12 envs × 150), mean latency ≤ 5.0 ms/step | K1 (Inference Latency Regression) |
| G2 | ≥ 1 of 12 envs solved (levels_completed increase > 0) | K2 (Zero Task-Solving Emergence) |
| G3 | mean per-step goal phase alignment delta Δ̄ν_goal > 0.0 | K3 (Directional Steering Failure) |
| G4 | multi-step horizon coherence Δ̄_Sagnac ≤ 0.050 | K4 (Waveform Coherence Degradation) |

## 3. Engine Architecture (`arc_f13_steering_engine.py`)

Tier 1 — Waypoint decomposition:
```
Ψ_waypoint = (1 − τ)·Ψ_t + τ·Ψ_goal, normalized to S^{D−1}, τ = 0.25
```
Tier 2 — Sagnac-guided macro-action beam search (depth K = 8):
```
J(a_{1:K}) = |⟨Ψ̂_{t+K}(a_{1:K}), Ψ_waypoint⟩| − α · Σ_{k=1..K} Δ_Sagnac(Ψ̂_{t+k}, Ψ_waypoint)
a_{1:K}* = argmax J;  commit a_t* = a_1*
```
Rollouts: sequential zero-trainable orthogonal `exp(D_a)Ψ̂_{t+k}` (seeded skew-symmetric generators, D=64 compact substrate, 8 actions).
Beam width B = 8 (directive specifies depth K=8; width pre-registered, deviation D3).

Tier 3 — Directional execution: commit first macro-action through the live Arcade adapter.

Tier 4 — Goal-progress valence plasticity (signed, directive literal):
```
Δν_t = |⟨Ψ_{t+1}, Ψ_waypoint⟩| − |⟨Ψ_t, Ψ_waypoint⟩|
M_{a_t} ← Normalize_L2(M_{a_t} + η_fast · Δν_t · Ψ_t),  η_fast = 0.05
```
Reuses `ActionPrototypeMemory` from F11 (`arc_f11_plasticity_engine`), `PatchIngress`/`sagnac_delta`/`_to_flat` from F10 (`arc_f10_live_engine`).

## 4. Pre-Registered Parameters (directive command, verbatim)

```
python HENRI V2/experiments/verification/arc_f13_steering_engine.py \
    --device cuda --steps-per-env 150 --seed 20260911 --horizon 8 \
    --tau-waypoint 0.25 --eta-fast 0.05 \
    --out-dir /tmp/henri_f13_steering/ \
    --receipt-out /tmp/henri_f13_steering/f13_gates_receipt.json
```
- 12 environments (F10 receipt cohort), 150 steps each = 1,800 steps.
- Flag: `HENRI_F13_STEERING=1` (default OFF, fail-closed).
- Seed 20260911; deterministic module-level seeded init (generators `seed+10`); `torch.manual_seed(seed)`.

## 5. Deviations (pre-registered, disclosed)

- **D1 — Goal source:** Ψ_goal = episodic first-frame observation wave (no Zone C DSN in the directive command; no approved secret channel). Same convention as F12 D1.
- **D2 — Tier-1 TimesFM-3:** the blueprint names TimesFM-3 multi-patch lookahead; the directive's executable formula is geodesic interpolation and the bounded command carries no TimesFM-3 flag/DSN. Implemented as written; TimesFM-3 module stays BLOCKED (standing §4).
- **D3 — Beam width B=8** (directive fixes depth K=8 only).
- **D4 — Sagnac penalty α = 0.05** (the directive's own validity threshold; term scaled to not dominate |cos| ∈ [0,1] over K=8).
- **D5 — G4 instrument:** Δ_Sagnac(k) = sagnac_delta(Ψ̂_{t+k}, Ψ_waypoint) along the committed macro-path; G4 = mean over all k across all steps.
- **D6 — Valence is signed** (directive literal): creep applies η·Δν·Ψ_t with signed Δν; creeps telemetry = count of applied updates (diagnostic only; G3 is the mean Δν_goal gate).
- **D7 — r_ext/progress accounting** carried from F11 D2 (`Δlevels_completed − 0.5·reset_penalty`) for progress telemetry only; selection is purely directional.
- **D8 — Compact scale:** D=64 verification substrate (production 8×65,536; blueprint scale), consistent with F10–F12.
- **D9 — Ledger alignment:** directive cites Record 1,079 = this directive's ingest event; F12 verdict is 1,078. Seal targets 1,080+.

## 6. Pre-Flight Kill Contract (K2 rule, before any live run)

C7-style synthetic in-sample steering contract: on a constructed favorable stream (state moving monotonically toward the waypoint), mean Δν_goal must be > 0.0. If the harness yields mean Δν_goal ≤ 0 on the favorable fixture, the harness is defective → fix and re-verify before the live gauntlet (no live run without this pre-flight pass).

## 7. Receipt & Verdict

- Receipt: `/tmp/henri_f13_steering/f13_gates_receipt.json` (schema `f13-steering.v1`), pinned SHA-256.
- Verdict vocabulary: `F13_LIVE_ENGINE_BLOCKED` (G1 false), `F13_GOAL_STEERING_VERIFIED` (all gates), `F13_GATE_{G2,G3,G4}_FAILED` (first failing gate, fail-closed precedence).
- Seal: `F13_GATES_VERDICT` ledger event with {schema, directive_sha256, git_sha, receipt_sha256}.
