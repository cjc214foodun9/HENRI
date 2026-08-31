# Carrier F11 — Closed-Loop In-Situ Reward Plasticity & Sub-Goal Ingress (Δν → Viscoelastic Creep): Pre-Registration

**Directive:** HENRI-DIR-2026-08-F10-POSTMORTEM-REWARD-PLASTICITY
**Directive file:** `Project_HENRI_F10_Post-Mortem_Audit___Closed-Loop_Reward_Plasticity_Directive.md`
**Directive SHA-256:** `73c4bc56c0af79f757e0a4f383d43aa026511942424c5709f005a43c51296d38`
**Branch:** `carrier/f11-reward-plasticity`
**Engine:** `HENRI V2/experiments/verification/arc_f11_plasticity_engine.py`
**Tests:** `HENRI V2/tests/contract/test_f11_plasticity_engine.py`
**Receipt schema:** `f11-plasticity.v1`

## 1. Ratification & Scope

Carrier F10 verdict `F10_GATE_G2_FAILED` (seal `#6562067c`, receipt `2afa5bcf…`) is RATIFIED by the directive. F11 closes the measured action-outcome disconnection: the exteroceptive reward channel Δν is coupled back into the action selector via (Tier 1) valence-weighted selection, (Tier 2) fast Hebbian action-prototype creep, and (Tier 3) anisotropic Langevin escape on negative valence.

## 2. Live-API Audit (OBSERVED 2026-08-31, remote probe, vast-5090)

- `arcengine.GameAction` members: **8** = `[RESET, ACTION1..ACTION7]` (directive says 7 — deviation D1: `M ∈ R^{8×65536}`).
- `game.step(action)` returns a pydantic observation with fields `{frame, levels_completed, win_levels, state, available_actions, ...}`. **No `reward`, no `done`, no `score` field exists** (directive says `step_result.reward` — deviation D2, disclosed).
- Therefore the exteroceptive valence channel is defined faithfully as:
  `Δν_t := (levels_completed_t − levels_completed_{t−1}) − 0.5 · [reset_penalty at t]`
  A level completion gives +1.0; a terminal reset (GAME_OVER or `None` step result) applies −0.5. This is the ONLY external progress signal the live API exposes, and it is exactly the signal F10's G4 measured as 0.0 (the engine never consumed it).
- Ledger state at execution start: 1,070 records (two `context_watchdog` records appended since the directive's "1,068"); F11 seals land at 1,071+.

## 3. Mechanism (per directive §3)

1. **Tier 1 — Valence-weighted selection (reward-augmented EFE):**
   `a* = argmin_a [ Δ_Sagnac(Ψ̂_{t+1}(a)) − λ_rew · R̂(s_t, a) ]`, `λ_rew = 2.0`.
   `R̂(s_t, a)` = per-action moving average of Δν history (decay 0.9), initialized 0.
   Candidate actions = the 8-member GameAction enum, masked to the env's `available_actions` when non-empty.
2. **Tier 2 — Fast-plasticity Hebbian creep:** `M_{a_t} ← Normalize_L2( M_{a_t} + η_fast · Δν_t · Ψ_t )`, `η_fast = 0.05`.
   `M ∈ R^{8 × 65536}` = per-action prototype matrix in GPU VRAM (float32). After every live step with non-zero Δν, the selected action's row creeps toward (positive valence) or away from (negative valence) the current wave state. `R̂` is derived from `M` cosine sim when history is sparse.
3. **Tier 3 — Anisotropic Langevin escape:** if `Δν_t < 0` and no positive Δν in the last 10 steps, inject `T_active = T_base + κ·max(0, −Δν_t)` (T_base = 0.15, κ = 0.5) as `sqrt(2·T_active·dt)` noise on the ingress projection before the next selection, breaking repetitive cyclic actions (directive: T=0.50 escape in 3 steps).

All three tiers are default-OFF behind `HENRI_F11_PLASTICITY=1`; the F10 engine (open-loop) remains the default path. Zero pretraining: M starts at zeros (R̂ = 0 ⇒ pure epistemic selection until first valence event).

## 4. Gates (directive §4, verbatim criteria)

| Gate | Criterion | Kill |
|---|---|---|
| G1 | 720 steps across 12 envs, latency ≤ 5.0 ms/step | K1 inference-latency regression |
| G2 | ≥ 1 of 12 live envs solved end-to-end (Score > 0.0) | K2 zero task-solving emergence |
| G3 | cumulative ΣΔν > 0.0 across cohort | K3 reward-coupling ineffectiveness |
| G4 | mean single-pass K=8 Sagnac ≤ 0.050 | K4 waveform-coherence degradation |

Verdict map: any G1 fail → `F11_LIVE_ENGINE_BLOCKED`; all pass → `F11_REWARD_LOOP_VERIFIED`; else first failing gate → `F11_GATE_GX_FAILED`.

## 5. Bounded Execution (directive §5 command, verbatim)

```bash
env HENRI_F11_PLASTICITY=1 PYTHONPATH='HENRI V2' /venv/main/bin/python 'HENRI V2/experiments/verification/arc_f11_plasticity_engine.py' \
    --device cuda --steps-per-env 60 --seed 20260909 --lambda-reward 2.0 --eta-fast 0.05 \
    --out-dir /tmp/henri_f11_plasticity/ --receipt-out /tmp/henri_f11_plasticity/f11_gates_receipt.json
```

12 envs, 60 steps each, seed 20260909, λ_rew 2.0, η_fast 0.05. Runs 1–N invalid launches preserved as evidence; only the valid run at the sealed candidate SHA counts.

## 6. Pre-flight kill (synthetic in-sample descent)

Contract test C7 asserts: on a synthetic stream with an unambiguous positive-valence action signal, the Tier-2 creep moves the action prototype toward the rewarded wave (cos ↑) and R̂ for the rewarded action increases — proving the Δν → M path is not a mock loop BEFORE the live gauntlet.

## 7. Deviations (disclosed, pre-seal)

- D1: M ∈ R^{8×65536} (live GameAction count = 8, incl. RESET), not 7.
- D2: Δν defined from `levels_completed` deltas minus reset penalty; the directive's `step_result.reward` field does not exist in the live arcade API (probed 2026-08-31).
- D3: ledger target 1,071+ (not 1,069+); watchdog records 1,068–1,069 exist.
- D4 (post-seal amendment, 2026-08-31): engine runs at the RATIFIED F10 scale — D=64 (num_blocks=8), so M ∈ R^{8×64}. The blueprint's 8×65536 applies at production scale (D=65,536); the F11 gauntlet at the ratified scale is the faithful F10-comparable carrier. `F11_SPEC_AMENDED` child event records this change with both spec hashes.
