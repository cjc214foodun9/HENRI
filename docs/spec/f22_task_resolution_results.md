# Carrier F22 — Task Resolution Results and Lessons

## Evidence

- Directive: `HENRI-DIR-2026-08-F21-1-POSTMORTEM-TASK-RESOLUTION`
- Directive SHA-256: `841c73a5b43058261ba31b0a17760f0674f8a3126e5a111ecb631f35666eaa49`
- Preregistration SHA-256: `045ab401d0bf79436bf15d31f36290c27ae2518ef8363900bb25dd0db4f22901`
- Code commit used for the live rerun: `f98330a505162c5d0d649c468f77f228c123cd32`
- Governance seal: `F22_GATES_VERDICT #0612ff513a2f40d4…`, ledger record 1,140
- Live receipt: 1,106 B, SHA-256 `22635fb2b7c2a6e22a0e10bf6dfe146532130517de585f871399ef1951d31001`
- Live log: 3,990 B, SHA-256 `e69b8062ef20b19aacdd861dc9511ccfeb04ca0beba34da89d61b51ee8e13ac`
- Runtime: Vast RTX 5090, `/venv/main/bin/python`, PyTorch 2.12.0+cu130, CUDA 13.0
- Remote worktree: `/tmp/f22-task-resolution-wt`, detached at `f98330a`

## Blocked attempt

The first 12-environment launch stopped after 900 steps when the Arcade API timed out while downloading a game asset. It returned `F22_ARCADE_MAKE_NONE`. This is **BLOCKED_INFRA**, not a task outcome. The log and receipt were preserved at `/tmp/henri_f22_resolution/` on the remote host.

All 12 game assets were then prefetched successfully through the real Arcade API. The same sealed bounds were rerun without code changes.

## Completed live rerun

Bounds: 12 named environments × 150 steps, seed `20260922`, horizon 8, omega bound `0.0982`, beta `0.015`, waypoint threshold `0.60`, Langevin temperature `0.50`.

| Gate | Criterion | Observed | Status |
|---|---:|---:|---|
| PG1 | capped-generator reconstruction ≥ 0.85 | 0.881334 | PASS |
| G1 | interactive latency ≤ 2.0 ms | 0.876707 ms | PASS |
| G2 | solved environments ≥ 1/12 | 0/12 | FAIL |
| G3 | waypoint valence ≥ +0.0200 | +0.000894 | FAIL |
| G4 | physical Sagnac loss ≤ 0.0500 | 0.806277 | FAIL |

Additional observed values: 1,800 steps, 56 resets, 13 waypoint advances, 56 Langevin escapes, 389 positive observed valence events, and zero levels completed in every environment.

Sealed verdict: **`F22_GATE_G2_FAILED`** by preregistered precedence.

## Lessons

1. **F21.1 substrate performance transferred.** Capped EDMD reconstruction remained above the F22 gate and live interactive latency remained below 2 ms.
2. **Dynamic waypoint plumbing engaged but did not solve the task.** The run advanced 13 waypoints, but observed post-action waypoint valence was only +0.000894 and the final alignment decreased from 0.591045 to 0.271577. Waypoint advancement is engagement evidence, not task capability.
3. **Internal predicted coherence did not transfer to external outcome.** Zero levels completed across 1,800 steps falsifies the assumption that a positive or stable internal operator score is sufficient for Arcade progress.
4. **G4 remains unresolved by the disclosed D=64 axiom substitute.** The measured physical loss 0.806277 is far above 0.05. The random seeded substitute is not evidence for the directive's claimed Zone C physical axiom coherence; a future carrier needs an authorized, dimension-compatible axiom source and a causal metric audit before retesting.
5. **Langevin reset noise did not produce measurable task escape.** 56 resets triggered the three-step escape mechanism, but no level completed. Do not increase temperature or duration without a new kill experiment and an external action-effect audit.

F22 is closed fail-closed. No promotion to `main` is authorized by this result.
