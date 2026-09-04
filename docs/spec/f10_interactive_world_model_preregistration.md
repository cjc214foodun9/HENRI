# Carrier F10 — Live Interactive Time-Series World Model (TimesFM-3 Continuous Ingress & Closed-Loop Active Inference): Pre-Registration

**Directive:** HENRI-DIR-2026-08-F9-1-POSTMORTEM-TIMESFM3-SYNTHESIS (18,535 B / `149bd93b02…`)
**Compiled by:** henri-arbiter (root holon), 2026-08-31
**Sealed:** `F10_PREREG_SEALED` (ledger record TBD)

---

## 1. Ratification and paradigm shift

The F9.1 verdict `F9_1_OPTIMIZATION_FAILED` (seal `#8a0cbcd8`, ledger 1,064, receipt `03cc2660…`) is **RATIFIED**. The frozen-bank optimization line is **permanently closed**:

- F4–F8.1: passive representation families, max CV 0.4617 (F8.1).
- F9 (Euclidean QR) / F9.1 (Riemannian Stiefel): in-sample CE stuck at ≈ 1.9189 ≈ ln 7; the objective cannot move.
- Data Processing Inequality: I(Ψ_bank; A) ≈ 0.0965 nats (F3 G4). No post-hoc optimizer recovers information filtered out at capture.

**Carrier F10 replaces bank training with LIVE environment interaction.** The engine queries the live Arcade environments (raw grids + reward deltas), patches the raw observation stream (TimesFM-3 pattern, p=32), and runs closed-loop active inference with Sagnac-guided horizon rollouts.

## 2. Hypothesis and mechanism

**HYPOTHESIS F10-H1 (falsifiable):** A live closed loop that patches RAW grid observations (not pre-embedded bank waves) carries a decodable action channel: in-sample CE on live observation patches descends below the ln-7 floor (G1-style metric), and task-level resolution emerges (G2) with positive exteroceptive progress (G4).

**Mechanism (Tier 1–4 of the directive):**

1. **TimesFM-3 continuous patch ingress (Tier 1).** Raw grid observation x ∈ {0..9}^(H×W) is flattened per-channel into contiguous patches p=32; each patch passes a residual MLP block (LayerNorm → linear → GELU → linear → residual) producing patch tokens t_j,v; 2D alternating spatio-temporal phase coupling forms the wave: Ψ_t = StiefelNormalize(W_patch · x̃_t + W_goal · Ψ_goal) ∈ S^(D−1). No pretrained TimesFM-3 weights are loaded (zero-pretraining invariant); the pattern is architectural only.
2. **Single-pass non-autoregressive horizon (Tier 2).** K=8 masked steps evaluated in one forward: Ψ̂_{t:t+K} = V W† (Ψ_t ⊛ A_{1:K}).
3. **Sagnac homodyne pruning + EFE selection (Tier 3).** Branch veto when Δ_Sagnac = 1 − |⟨Ψ̂_{t+1}, Ψ_axiom⟩| > 0.35; action a* = argmin_a G_EFE(Ψ̂_{t+1}(a), Ψ_goal).
4. **Online valence-anchored plasticity (Tier 4).** ΔW = −η ∇F_EFE · I(Δν > 0) + √(2 T_active) dW; parameter creep only on positive exteroceptive progress.

## 3. Gates (directive §4, verbatim)

| Gate | Criterion | Kill |
|---|---|---|
| G1 | 60 interactive steps in all 12 environments, no crash, no static fallback loop; step latency ≤ 50 ms | K1 (Live Interaction Pipeline Defect) |
| G2 | Solve ≥ 1 of 12 live environments end-to-end (Score > 0.0%) | K2 (Zero Task-Solving Emergence) |
| G3 | Mean single-pass Sagnac loss across horizon K=8 satisfies Δ̄ ≤ 0.35 | K3 (Multi-Horizon Waveform Dispersion) |
| G4 | Cumulative exteroceptive progress rate ΣΔν > 0 across cohort | K4 (Zero Exteroceptive Progress Coupling) |

**Verdict mapping (pre-pinned):** all G1–G4 pass → `F10_LIVE_LOOP_VERIFIED` (capability evidence: task scorecard). Any gate fails → `F10_GATE_<N>_FAILED` with the specific kill; G1 fails → `F10_LIVE_ENGINE_BLOCKED` (pipeline defect, not science).

## 4. Resource and runtime bounds

- Live Arcade API (12 envs), 60 steps each, seed 20260908, K=8 horizon, p=32 patches.
- No frozen-bank training; no dense [D,D] tensors (factorized V,W† only; no-dense-allocation contract).
- Default-OFF env guard: **`HENRI_F10_LIVE=1`** (single aligned name, spec ≡ impl ≡ launcher).
- Receipt schema: `f10-live-engine.v1`.

## 5. Deviations and disclosures (pre-seal)

1. Directive's forensic fold table env names mismatch the sealed F9.1 receipt (prose defect; receipt values authoritative).
2. Directive says "Record 1,065+"; live ledger is 1,065; next seal lands 1,066+.
3. The live Arcade API requires production credentials / network reachability; if the live environment is unreachable or the public Arcade exposes no examples (BLOCKED_NO_PUBLIC_DEMOS), G1 is `BLOCKED_INFRA` — preserved as evidence, never faked.
4. G2's "Score > 0.0%" uses the Arcade `levels_completed` external outcome only.

## 6. Pre-registered kill experiments

- **K1 (cheap):** if 12-env live engagement fails before 60 steps on the first env → engine defect, fix launcher, rerun SAME candidate.
- **K2 (cheap):** synthetic separable live-patch fixture (class-conditional means well-separated) must descend in-sample (CE ≪ 0.5, P@1 → 1.0 within 5 epochs). If the engine cannot pass this fixture locally, a live K1/G1 result would be guaranteed — the synthetic test is the decisive pre-flight.
- **K3/K4:** single-pass Sagnac coherence and progress rate measured on the LIVE run only.
