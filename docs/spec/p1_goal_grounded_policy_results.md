# Carrier P1 Results: Goal-Grounded Policy Steering

**Carrier:** `P1_GOAL_GROUNDED_POLICY_STEERING` (29th carrier)
**Packet:** `Carrier_P1_SpecContract___Alignment_Probe.md` — SHA-256 `06e667c33134f82924c0d9500dfa8c8ee8ab5c1e2dd6484478296b4161fd3989` (18,819 B / 388 lines)
**Prereg:** `docs/spec/p1_goal_grounded_policy_preregistration.md` — SHA-256 `f4f73d54…` (sealed @1,206 `a38874e5`)
**Branch:** `feat/carrier-p1-policy-grounding` — commits `7c52a77` (bank probe), `dc44ec4` (surgical patch + prereg)
**Causal parent:** G7 `G7_AFFORDANCE_FIT_COLLAPSE` `#25e96e09` @1,198 (27th falsification)

## Execution record (OBSERVED)

- Bank alignment probe (step 1): **PASS** — sealed @1,205 `da597d5f`; 1,536 rows = 1,536 jsonl lines; 12 envs contiguous; terminals exact; 0 misalignments; 0 terminal discrepancies.
- CUDA contract suite at exact SHA `dc44ec4` on Vast: **24/24 passed** (P1 6/6 + G7 18/18). Local: P1 6/6; G-series regression 63 passed / 3 skipped.
- Live gauntlet: PID 902217, EXIT:0, wall 1,179 s, GPU released, 12 envs × 150 steps = **1,800 steps**, 17 resets, 1,783 affordance updates.
- Receipt: `/tmp/henri_p1_goal_steering/p1_gates_receipt.json`, SHA-256 (remote) `…` (verified byte-identical local copy in Temp).

## Pre-flight (3rd consecutive full clearance)

PG1 global 1.0; PG1a a0–a6 subset AUC all 1.0 (N=256); PG2 norm drift 1.19e-7 ≤ 1e-6; PG3 pass; C2 dense α == 0.0 exactly; τ_a all 0.05 (floor); subset 256 rows.

## Live gates

| Gate | Metric | Threshold | Measured | Result |
|---|---|---|---|---|
| Engagement | p1_score_calls | > 0 | **1,800 / 1,800** | ✅ PASS |
| Engagement | mean potential drops | action-discriminating | **[+0.048, +0.046, +0.017, +0.018, +0.021, +0.026, −0.027]** (a6 negative) | ✅ PASS — first measured goal-term discrimination in the G-series chain |
| LG2 | solved levels | ≥ 1 | **0 / 12** | ❌ FAIL — seal basis |
| LG1 | mean Δν (waypoint) | ≥ 0.05 | 0.0 (unmeasurable — see defect) | ⚠ DISCLOSED, not evidence |
| LG3 | local kernel latency | ≤ 2.0 ms | 10.86 ms (CUDA-event) | ⚠ PERF FLAG, not seal basis |
| — | g4 affordance mean | ≤ 0.05 | 0.0015 | ✅ PASS |

## Disclosures (NOT seal basis)

1. **Inherited Δν measurement defect (G4 → G7 → P1):** in `arc_g4_aligned_engine.run_gauntlet` the post-step alignment `c_next` is computed from the **pre-step `psi64`** (never re-encoded from `frame_next`), so `c_next == c_t` every step: `mean_delta_nu_wp` is structurally 0.0 and `creeps` can never fire. Every "mean_delta_nu_wp: 0.0" in the G4–G7 receipts is this artifact. LG1 as inherited cannot measure progress; this does not alter G7's seal (which also rested on 0 solved / 0 waypoint advances — genuine arcade outcomes), it removes Δν as a valid evidence line retroactively. Fix requires a measurement carrier (re-encode `frame_next` → `psi64_next`), NOT a policy change.
2. **LG3 latency:** 10.86 ms local score-path latency (CUDA events) exceeds the 2.0 ms gate due to 448 tiny per-block einsum launches. Perf-only finding; prereg de-rates LG3 to a flag. The fused 7-action candidate-projection kernel (packet Action-Plan step 3) is the follow-up, not part of this carrier.
3. Emitted code verdict `P1_GATE_LG3_LATENCY_FAILED` fires first by fail-closed precedence; the binding seal basis is LG2.

## Verdict

**`P1_GATE_LG2_SOLVED_FAILED`** — 28th sealed falsification / 29th closed carrier. **FAIL_CLOSED. No retry** (prereg kill criteria: LG2 failure seals; hyperparameters frozen this carrier).

## Bisection update (29 carriers: 28 falsifications, 0 solved)

- Representation/affordance fit: SOLVED (3 consecutive pre-flight clearances; PG1 all 1.0).
- Goal-signal engagement in the policy: **SOLVED (new)** — the action-conditioned potential drop ΔV(a) measurably discriminates actions with correct sign structure (+ for 5 actions, − for a6), 100% of live steps routed through the P1 scorer.
- Live action → external outcome coupling: **UNSOLVED (persists)** — 0/12 solved despite a goal signal that now demonstrably steers action scores.

**Narrowed inference:** the remaining gap is not "the goal term is inert" (falsified this carrier) and not "affordance representation" (3× cleared). The open hypotheses for a successor carrier: (a) the goal semantics — a bank terminal-frame wave may not be the correct steering target for level completion under the arcade contract; (b) the action→environment interface (whether the selected action's payload/coordinate semantics actually move the level toward the goal); (c) ΔV scale — mean drops 0.017–0.048 modulate j by ≤5%, possibly too weak against π^H.

## W0 (WavePacketPathSearch)

**STAYS GATED.** `P1_POLICY_GROUNDING_VERIFIED` was NOT achieved; C3 auto-gate not triggered. No W0 work performed.
