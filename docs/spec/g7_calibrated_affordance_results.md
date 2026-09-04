# Carrier G7 Results — Calibrated Stratified Affordance (Launch #1 + #2)

**Directive:** `Carrier_G7_Master_Directive___G6_Post-Mortem.md` (`HENRI-DIR-2026-09-V3-CARRIER-G7-CALIBRATION-EXPANSION`, SHA `24b7665d…`, packet `#d2cbf9b5` @1,190)
**Launch-2 protocol:** `Carrier_G7_Launch__2_Master_Protocol___Clearance_Audit.md` (`HENRI-SPEC-2026-09-V3-CARRIER-G7-LAUNCH-2-CANONICAL`, SHA `816dbb0e…`)
**Prereg:** `docs/spec/g7_calibrated_affordance_preregistration.md` (SHA `02083db5…`, commit `ad47d7d`, sealed `#a795c151` @1,191)
**Branch:** `feat/carrier-g7-calibrated-affordance` @ `dce55af` (engine `arc_g7_calibrated_engine.py` + contracts `test_g7_calibrated_engine.py` @ `537f1ef`, harness fix @ `dce55af`)
**Env:** Vast RTX 5090 (vast-5090), CUDA 13.0 / PyTorch 2.12, seed 20260930, 12 envs × 150 steps, bank pinned `9e3c01b4…`

## Verdict — SEALED FAIL-CLOSED (27th falsification / 28th carrier)

`G7_AFFORDANCE_FIT_COLLAPSE` — sealed `#25e96e09…` @1,198 (`CARRIER_EXECUTION_VERDICT`, parent `#75873ea1` launch-#1 BLOCKED_INFRA @1,193). **W0 STAYS GATED.** No promotion to `main`.

## Launch #1 — BLOCKED_INFRA (zero steps, no verdict)

`#75873ea1` @1,193: PG1 pre-flight PASSED (first full clearance ever) then env #10 `bp35` transient `three.arcprize.org` read-timeout → `arcade.make()` → None → inherited `G4_ARCADE_MAKE_NONE`. Receipt-clobber bug exposed (`result.update(base_result)` zeroed `steps_done`) → fixed in `dce55af` (`finalize_receipt`, live fields win) + regression test. Startup-refusal rule: record + relaunch, NO verdict, NO quarantine.

## Launch #2 — LIVE GATE EVALUATION (first in project history)

Receipt SHA `a95e4ee6…`, wall 1,147 s, exit 0, scorecard `42e7652e`.

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| PG1 global (subset N=256) | ≥ 0.9000 | 1.0 | ✅ |
| PG1a a0–a3 (dense) | ≥ 0.9500 | 1.0 / 1.0 / 1.0 / **1.0** | ✅ (a3 FIXED vs G6 0.9412) |
| PG1a a4 (bridge) | ≥ 0.9500 | 1.0 | ✅ |
| PG1a a5, a6 (a6 bridge) | ≥ 0.8800 | 1.0 / 1.0 | ✅ |
| PG2 norm drift | ≤ 1e-6 | 1.19e-7 | ✅ |
| PG3 CPU==CUDA | identity | pass | ✅ |
| C2 α dense | == 0.0 | true | ✅ |
| Live steps | — | **1,800** (12×150) | ✅ executed |
| Affordance updates | > 0 | 1,783 | ✅ engaged |
| Envs solved | ≥ 1 | **0** | ❌ |
| Waypoint advances | ≥ 1 | **0** | ❌ |
| Mean Δν (waypoint) | ≥ 0.015 | **0.0** | ❌ |
| Latency note | — | 314.18 ms | ⚠️ inherited G1 constant (2.0 ms) measures end-to-end incl. remote arcade step; disclosed, NOT seal basis |

## Mechanism findings

- **Representation/affordance-fit class: SOLVED (twice).** PG1 global + all per-action subset AUCs = 1.0 across two consecutive runs (launch #1 pre-flight and launch #2). G6's a3 kill was confirmed finite-sample subset quantization: N=256 (1/32 step) resolved it; full-bank a3 0.9775 unchanged (validated the estimator-resolution hypothesis, not τ_a — per-action AUC is rank-invariant to the monotone τ_a transform).
- **Live transfer: UNSOLVED (0/12 envs, 0 solved levels, 0 waypoint advances across 1,800 real arcade steps).** The live action loop executes, learning engages (1,783 affordance updates, 17 resets, 314 ms/step incl. remote calls), yet no level completes. This is a genuine live dynamical failure per the Launch-2 protocol taxonomy → FAIL_CLOSED (no retry; infra was healthy — 12/12 warm cache, no timeouts).
- The falsified boundary is now precise: **wave representation → action affordance ranking → external outcome transfer**. Pre-flight says the affordance operator separates moving from non-moving bank transitions perfectly; the live loop cannot convert that separation into level-solving actions.
- Latency disclosure: `G7_GATE_G1_FAILED` was emitted by the inherited G1 `G1_LATENCY_MS=2.0` constant against 314 ms end-to-end (incl. remote `game.step()`). This constant is not in the G7 prereg vocabulary; the seal basis is the zero-outcome live result, not latency. Harness-naming defect recorded for a future bounded fix (a real planner-local latency gate must exclude network I/O).

## Chain state

28 carriers, 27 sealed falsifications (G7 = 27th), 0 solved envs. Ledger head @1,198 (`#25e96e09…`, 1,199 records verified); events @1,190 packet, @1,191 prereg seal, @1,193 launch-#1 BLOCKED_INFRA, @1,197 G7_LAUNCH2_COMPLETE (`#cc06ce70…`), @1,198 verdict. G7 results doc committed on branch tip; launch-#1 evidence preserved under `/tmp/henri_g7_calibrated/`, launch-#2 under `/tmp/henri_g7_calibrated_l2/`.

## Next levers (NOT authorized — require new directive)

1. Live-policy carrier: action selection over the now-verified affordance operator (the fitted transitions/top-k masks are bank-verified; the open problem is converting per-action affordance scores into level-solving action sequences under the arcade API contract).
2. Waypoint/Δν carrier: why the waypoint channel never advances on live frames (0/1,800) while bank kinematics compile cleanly.
3. W0 (WavePacketPathSearch planner wiring) remains gated — requires a FULL live-pass carrier first.
