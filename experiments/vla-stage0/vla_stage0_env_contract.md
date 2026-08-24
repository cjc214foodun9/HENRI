# System-1 Stage-0a — VLA Dynamical Substrate Environment Contract

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding**
**Upload audited:** `HENRI_v0.6.4_Premise_Audit___Stage-0_VLA_Substrate_Blueprint.md`
sha `514cc3a0…` (1302 B) — **VERIFIED**: correct restatement of the v0.6.4
premise-audit conclusion (transition pairs vs scalar labels). No new claims.

## Dispositions

| Claim | Disposition |
|---|---|
| Static verifier labels cannot form cross-covariance | VERIFIED (corpus #21, v0.6.4 audit) |
| Real substrate needs (x_t, x_{t+1}) transition pairs | VERIFIED (corpus #21/#22) |
| Stage-0 integration = new architecture carrier | ACCEPTED (Reference 3: stage 0a/0b/0c) |
| UWE on S^(D-1) at D=65,536 | **CONFLICTS_WITH_LIVE_CODE** — corpus #22 asserts 65,536-D authorized boundary; Reference 3 + live code + prior audits (v0.6.0/0.6.1/0.6.2/v0.6.3) use `[1,16,384]` / `d_slot=384`. **No new tensor family.** Encoder (Stage-0b) uses the live boundary. |
| "Unit-modulus error < 1e-6" | **CONDITIONAL** — valid ONLY for elementwise unit-modulus complex torus \|z_i\|=1; NOT for unit-norm sphere ‖z‖₂=1. Stage-0b contract will state which geometry is implemented. |

## Environment grounding (OBSERVED)

- gymnasium **1.3.0** on local `/c/Python314` and remote `/venv/main` (Vast 5090).
- `CartPole-v1`: obs `Box([-4.8,-inf,-0.4189,-inf],[4.8,inf,0.4189,inf],(4,),float32)`;
  actions `Discrete(2)`; `reset(seed=)` supported; `ResetNeeded` raised if
  `step()` before `reset()`. `max_episode_steps` configurable (default 500).
- "Grid-Gym": **BLOCKED_MISSING_PREMISE** — no canonical package/env ID
  identified. CartPole-v1 ONLY for the first carrier.

## Stage-0a contracts (this carrier — wrapper + provenance ONLY)

| # | Contract |
|---|---|
| C1 | Real tuple provenance: (s_t, a_t, r_t, s_{t+1}, terminated, truncated, episode_id, step_id) appended per step; **BOTH `obs_t` and `obs_next` persisted with hashes**; chain continuity `record[t].obs_next == record[t+1].obs_t` asserted |
| C2 | Same pinned seed + same action prefix → byte-identical traces |
| C3 | Different seeds → non-vacuous difference |
| C4 | Action validated against Discrete(2) before env.step |
| C5 | Terminal/truncated episodes reject further steps until reset |
| C6 | No state leakage across episodes; reset via `env.reset(seed=...)` ONLY; never mutate `env.unwrapped.state` as the reset mechanism |
| C7 | Raw observations hashed per transition |
| C8 | No synthetic states (`torch.randn` absent); no private-state mutation |

## C1 provenance correction (2026-08-24, governance `stage0a-provenance-correction-20260824-001`)

The ORIGINAL Stage-0a serialized records contained ONLY `obs_next` + `obs_next_hash`;
`obs_t` was never persisted, so C1 (tuple provenance) was NOT demonstrated
despite C2–C8 passing. Per the skill pitfall (a prose contract naming fields
absent from emitted telemetry is FALSIFIED), the wrapper was patched to persist
`obs_t` + `obs_t_hash` and assert chain continuity. Re-run: C1–C8 ALL PASS.
- Corrected provenance: `vla_stage0_provenance.json` sha `eb1a061c…`
  (8 transitions, seed 4242; obs_t present, chain continuity holds).
- Original preserved: `vla_stage0_provenance_legacy.json` sha `9555c13d…`.

Deterministic branch reconstruction (Reference 3): branch state =
`reset(seed)` + `replay(action prefix)`; verify byte-identical trace.
Direct internal-state cloning is CartPole-specific and CONDITIONAL.

## Gated downstream (NOT this carrier)

- **Stage-0b** frozen encoder on live `[1,16,384]` / `d_slot=384` boundary
  (state geometry declared explicitly; no 65,536-D family). Default OFF.
- **Stage-0c** R-EDMD adapter: action-conditioned lifted transitions
  x_t=ψ(s_t,a_t), y_t=ψ(s_{t+1}); η=0 byte-identical baseline; bounded
  forgetting factor/condition number/spectral radius/update norm/rank/
  runtime/memory; branch-isolated fast weights with rollback; prediction
  error vs no-learning and fixed-linear baselines; `O(r²·D)` NOT claimed
  until measured profiler telemetry establishes it.
- **Evaluation boundary:** CartPole realizes C1/C4 and is a REAL dynamical
  substrate. It does NOT create vision, language conditioning, or an
  Artificial Analysis capability path. Verdict for Stage-0a:
  `DYNAMICAL_SUBSTRATE_VERIFIED` — NOT "VLA integrated", NOT benchmark progress.
  No heldout, no CUDA promotion for the wrapper contract; deterministic CPU
  tests are the appropriate initial evidence.
