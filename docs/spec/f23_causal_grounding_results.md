# Carrier F23 — Online Causal Calibration & Semigroup Axiom Grounding — Results

**Directive:** HENRI-DIR-2026-08-F22-POSTMORTEM-CAUSAL-GROUNDING
(`9cafa2a3fcc457b0784cc9d9e321d7f02af8a3ae65e6a1696921673a65ec106c`, 20,587 B, 244 lines)
**Branch:** `carrier/f23-causal-grounding` @ `46706485ce788077858fbf812004078c8bc9015c`
**Prereg:** `docs/spec/f23_causal_grounding_preregistration.md` (`ffd16a9f…`, sealed `#9ae95dc2…`)
**Receipt:** `f23_gates_receipt.json` SHA-256 `90e3b094c4531f746e26ab9a32b76bc1ead1afa15cfa89c3ca2a8967dd2f07cd`
**Verdict:** `F23_GATE_G1_FAILED` (fail-closed), sealed `#c2ee09e2f547db6f…` (ledger record 1,145)
**Run:** 12 envs × 150 steps = 1,800 live Arcade steps, seed 20260923, RTX 5090 CUDA 13.0

## Gate results

| Gate | Threshold | F23 measured | Status |
|---|---:|---:|---|
| PG1 | ≥ 0.8500 | 0.893474 | PASS (pre-flight) |
| G1 | ≤ 2.00 ms | **2.539839 ms** | FAIL |
| G2 | ≥ 1/12 | **0 / 12** | FAIL |
| G3 | ≥ +0.0150 | **−0.000384** | FAIL |
| G4 | ≤ 0.0500 | **0.923024** | FAIL |

## Mechanism engagement (measured, not inferred)

- Calibration fired on **1,660 / 1,660** non-terminal steps (fully engaged; not inert).
- Semigroup axiom synthesis correct as math: leading eigenvalue **0.999393**.
- Stall hysteresis: **47,833 penalty events** ≈ 26.6 per step → persistent near-identical-state signature.
- Resets 140 (F22: 56); waypoint advances **1** (F22: 13); creeps 440; waypoint alignment collapsed **0.4272 → 0.0879**.

## Falsification

The F23 hypothesis — in-situ rank-1 Stiefel calibration + semigroup stationary axiom
+ causal-horizon stall memory closes the sim-to-real gap and yields task resolution —
is **FALSIFIED**: 0/12 solved, G3 valence went *negative* (worse than F22's +0.000894),
G4 0.9230 (worse than F22's 0.8063), G1 regressed 2.54 ms (F22: 0.877 ms). The three
directive upgrades are insufficient as a set; calibration is causally engaged yet
task-level resolution does not transfer (21st sealed carrier, 0 solved envs).

## Root causes (measured/derived)

1. **G1 regression — stall-memory Python loop:** `score_all_actions` iterates up to 32
   stall-memory entries per step with per-entry `.to(device)` + cosine ops. The C9
   no-for-loop guard only covers `step_once`, not `score_all_actions`. This is the
   latency delta (0.877 → 2.54 ms) and an implementation-level cause of G1 failure.
2. **G4 degenerate axiom direction:** stalled/no-movement transitions dominate the
   bank, so the average operator T̄ ≈ identity; its leading eigenvector is an
   arbitrary direction in a degenerate eigenspace (ev 0.9994 confirms near-identity),
   not a direction on the live state manifold → Sagnac 0.923.
3. **G3 negative valence:** post-action frames drifted *away* from the waypoint
   (mean −0.000384). Calibration rotated operators toward local no-movement
   transitions, not toward waypoint-advancing motion; alignment collapsed.

## Boundary

PG1 healthy. No promotion to `main`. Carrier closed fail-closed.

## Next (NOT directed)

Task-level causal action modeling: action-conditioned transitions that distinguish
moving vs blocked states and target waypoint-relevant displacement. Do not re-open
the three F23 upgrades without materially new evidence.
