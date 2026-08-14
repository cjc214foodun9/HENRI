# Phase 8 — Batched Navigation Swarm: Pre-Registered Design Packet

**Phase:** 8.0-batched-nav-swarm
**Status:** PRE-REGISTERED (design only — no implementation yet)
**Date:** 2026-08-13 (UTC)
**Arbiter:** HENRI development arbiter
**Base commit:** `ff8bfa00e0e5a5efc99f42978c872e13af4b8efe` (`phase/7.9f-curriculum-replay`, un-promoted)
**Branch:** `phase/8.0-batched-nav-swarm` (created from base)
**Flags (all default OFF):** `HENRI_ARC_BATCHED_NAV_SWARM=1` (master switch), `HENRI_ARC_SWARM_ARM=a|b|c|d`, `HENRI_ARC_SWARM_B=64|256|512|1024`, `HENRI_ARC_SWARM_DIVERSITY=1`, `HENRI_ARC_SWARM_PT_REPLICAS=4`, `HENRI_ARC_ESS_THRESHOLD=0.5`; existing `HENRI_ARC_ACTION_PAYLOADS=1`, `HENRI_ARC_EGRESS=1` required for payload/egress arms.
**Evidence class for all outcomes:** `diagnostic_only=true`, `score_eligible=false` until a separate promotion gate is approved.

---

## 1. Background and sealed evidence (inputs to this design)

| # | Fact | Class | Source |
|---|------|-------|--------|
| E1 | 7.9e probe: 4/4 envs `BLOCKED_NO_PROGRESS_EVENTS`; 960 branches, 90,240 steps, 0 progress events | OBSERVED | p79e receipt `829597c7…`, aggregate `38fae7a7…` |
| E2 | 7.9f matrix at `ff8bfa0`: 12/12 discovery envs `BLOCKED_NO_PROGRESS_EVENTS`; 2,989 matched branches, 65,041 steps, 0 strict scorecard progress events; SANS rows = 0; held-out sealed | OBSERVED | p79f receipt `4246443e…` (aggregate) |
| E3 | K(Ψ) descriptors across 7.9f discovery envs ranged 0.451–0.906 (descriptor only, not progress) | OBSERVED | p79f per-env JSONs |
| E4 | ARC Arcade exposes `examples: None`, `demonstrations: None` on g50t, dc22, ls20, lp85, cn04, s5i5 | OBSERVED | live probe 2026-08-14 |
| E5 | `TaskConditionedInductiveEncoder` does not exist in the live tree (0 grep hits); production already has `compile_task_functor` (`production_arc_run.py:77,630`) which blocks target-scored MCTS with `OBSERVED_TEST_TARGET_UNAVAILABLE` (~:652) | OBSERVED | live grep |
| E6 | `HenriSwarmOrchestrator` (`darwinian_phase_swarm.py:293`) is production-wired: `production_arc_run.py:1030 orch.plan_action(state_wave, boundary_batch, top_k=4, return_chosen=True)`; `GapJunctionSwarmSyncytium` (:84) = 1,024 experts, `[1024,16,65536]` projections; `ScaleFreeGraphConstructor` (:39); `generate_colored_langevin_noise` (:20); EFEPlanner (`efe_planner.py:187`) is `orch.planner` (`select_action`, `train_transition_batch`, `infer_goal_from_preferences` at :798) | OBSERVED | subagent audit (grep/read, read-only) |
| E7 | `CUDAGraphMCTSTreePool` (`henri_gpu_crystalline_active_inference.py:60,173`), `henri_fused_triton_cuda_graph_runner.py` (`CUDAGraphBatchedUnbinderRunner` :87), `WaveJEPA` (`wave_jepa.py:21`) are benchmark-only/archived; `henri_batched_cuda_graph_runner.py` does not exist | OBSERVED | subagent audit |
| E8 | Production egress = `HENRIUnifiedEgressTransducer` (`henri_decoder.py:462`) wrapping `HENRINeuralEgressUnbinder` (:58); D=65,536 → d_hidden=2,048 → |V|=32,000; used at `production_arc_run.py:1108–1135` | OBSERVED | subagent audit |
| E9 | Particle state envelope: 1 particle at D=65,536 FP32 = 256 KiB; B=1,024 = 256 MiB per full state pass; ~2.1 TB/s ⇒ ~122 µs floor; workload is bandwidth-bound, not FLOP-bound; attained fraction of peak bandwidth decreases as peak bandwidth rises | DERIVED + OBSERVED (arXiv:2605.30571) | this packet |
| E10 | Replicated 50M-param models ×1,024 (FP16) ≈ 102 GB — infeasible; shared weights + batched states is the only feasible topology | DERIVED | this packet |

### 1.1 Data boundary (frozen)

- **Split** (pre-registered 2026-08-13, seed `20260813`, salt `p79f-split-v1`): discovery 12 = cn04, ka59, g50t, sb26, ar25, lp85, dc22, m0r0, bp35, ls20, re86, ft09; held-out 4 = s5i5, r11l, cd82, lf52. Frozen; **post-outcome selection is prohibited** (the earlier "top-4 K(Ψ) ≥ 0.80" proposal is rejected by this clause).
- **Demonstration boundary:** no authorized demo ingress exists (E4, E5). `W_task` compilation from demos remains `BLOCKED_NO_DEMONSTRATIONS`. Environment implementation files are evaluation machinery, not demonstrations.
- **Score signal:** strict `levels_completed` increase via `arc_scorecard_delta.detect_level_progress` only. K(Ψ), EFE, entropy, Sagnac delta, frame-delta, cursor, ESS, action entropy = descriptors only.

---

## 2. Hypothesis (mathematical statement, falsifiable)

**H1 (compute):** A shared-weight, batched particle navigation over the frozen production wave policy can saturate the RTX 5090: at B=1,024, achieved bandwidth fraction ≥ 0.60 of peak (~2.1 TB/s) with CUDA-graph capture of the rollout kernel, and planning steps/s ≥ 1.2× the per-particle cost of arm A.

**H2 (task):** Given zero external progress at 1 particle per decision (E1, E2), scaling the *number of candidate continuations* B per decision with diversity controls (ESS-triggered resampling, repulsion, parallel temperature ladder) strictly increases aggregate `levels_completed` on the frozen discovery split, per matched environment-step budget, against arm A (top_k=4) and arm D (random stratified control).

**H3 (diversity necessity):** Arm C (with diversity controls) ≥ Arm B (without) on aggregate `levels_completed`; if B ≡ C (no gain), the diversity machinery is `FALSIFIED` as a mechanism (batching alone carries the effect or nothing does).

**Kill predictions (pre-registered):**
- K1: bandwidth fraction < 0.60 at B=1,024 after CUDA-graph capture ⇒ **KILL_COMPUTE_UNDERSATURATED** (compute hypothesis falsified; no env-semantics work proceeds).
- K2: sustained ESS < 0.5 (or action-entropy collapse to ≤ 1 distinct legal action per decision) under arm C ⇒ **KILL_COLLAPSE** (diversity controls inert).
- K3: zero strict scorecard progress events across all 12 discovery envs in arm C ⇒ **BLOCKED_NO_PROGRESS_EVENTS** (same sealed verdict as 7.9e/7.9f; no held-out, no SANS, no promotion).
- K4: any arm rc ≠ 0 or aggregate `env_count != 12` ⇒ **BLOCKED_INFRASTRUCTURE** (fail closed; no per-arm science claims).

---

## 3. Mechanism

Each decision at state wave Ψₜ (continuous UWE, `[num_blocks, 8]`, S^{D-1}):

1. Generate B particle states: `Ψ_b = normalize(Ψₜ ⊕ ε_b)` with colored Langevin noise (existing `generate_colored_langevin_noise`) and (arm C only) temperature ladder `β_b ∈ {1, β_max^((b-1)/(B-1))}` per particle.
2. Batch through the **shared frozen** policy stack: EFEPlanner scoring `[B, …]` (production `LAMBDA_CONSTRAINT_MAX=5.0`, `resolve_spatial_basis()`, planner frozen `orch.eval()`, no dropout) → per-particle logits.
3. Mask to the environment-legal action set (versioned action vocabulary + `ActionEgressVocabulary`/`decode_action_egress`); coordinate actions get payloads via the production `step_with_payload` path (`(GameAction, data)`, not bare enums).
4. (Arm C) ESS check after scoring: `ESS = 1/Σ w_b²`; if `ESS < HENRI_ARC_ESS_THRESHOLD`, resample with repulsion `+η Σ_{j≠b} exp(−‖Ψ_b−Ψ_j‖²/τ_c)`; PT exchanges accepted by Metropolis ratio.
5. **Select** one candidate per decision for env stepping: best-scored particle (arms B/C) or EFE top_k=4 (arm A) or uniform random legal (arm D). Env stepping stays **sequential** (single Arcade instance per branch); planning/scoring is GPU-batched.
6. **Matched counterfactual discipline (unchanged from Reference 3):** reset → identical prefix P=4 → verified branch-state hash → vary the first complete action `(GameAction, data)` → same frozen continuation policy → compare strict external outcomes. Unmatched states counted; non-reproducible resets fail closed `BLOCKED_INFRASTRUCTURE`.
7. **SANS rows:** trainable row = `(hidden, action_idx, delta_nu)`, `hidden = unbinder_hidden(transducer, flatten_uwe(encode(grid), d_model))` (normalize → down_proj → layer_norm → GELU, `[d_hidden]` float32 CPU); action label = step action at the horizon state; persisted losslessly as `hidden_*.pt` + immutable JSON manifest (`SANS_ROW_SCHEMA_ID="henri.sans-row.v1"`); split-filtered; held-out never counted; no valid hidden ⇒ `NO_HIDDEN_FEATURE`, never admitted.

### 3.1 Data path

```text
arcade.make(full_game_id) → reset → P=4 prefix (verified frame hash)
→ per-particle state waves [B, num_blocks, 8] → shared frozen EFE/transducer (GPU batch)
→ legal-masked logits → (GameAction, data) payloads → step_with_payload (sequential per branch)
→ scorecard levels_completed (detect_level_progress) → matched-branch bookkeeping
→ per-env immutable JSON → aggregate (env_count == 12, DONE only rc=0) → SHA-256 ledger
```

---

## 4. Arms (pre-registered, all on frozen discovery split)

| Arm | Policy | B | Diversity controls | Budget |
|-----|--------|---|--------------------|--------|
| A | Production EFE, top_k=4 (control; unchanged path) | 4 candidates | none | matched total env steps |
| B | Batched EFE particles, shared frozen weights | 64 → 256 → 512 → 1,024 (sweep) | none | matched total env steps |
| C | B + ESS resampling + repulsion + PT ladder | 1,024 (fixed after probe) | yes | matched total env steps |
| D | Uniform random legal action (positive control) | n/a | n/a | matched total env steps |

- **Matched budget rule:** total environment steps per arm = 12 envs × 60 rounds × 21 steps/round (4-prefix + 1 candidate + 16 continuation) ≈ 15,120 steps/arm (7.9e/7.9f scale). GPU planning cost may differ by arm — that is the measured variable, not a license to reduce env steps.
- **Per-arm telemetry isolation:** sequential GPU scheduling, no concurrent suites (GPU-exclusive rule). Record pre-launch proc count; require zero competing jobs.
- **Minimum valid branches:** per env ≥ `MIN_VALID_BRANCHES=40`; below ⇒ `INCONCLUSIVE_SPARSE_OUTCOME` (fail closed, never "no causal information").
- **Statistical floor:** acceptance requires aggregate progress events ≥ 5 with ≥ 2 distinct envs; fewer ⇒ `INCONCLUSIVE_SPARSE_OUTCOME`.

---

## 5. Compute-envelope probe (cheapest kill experiment — step 0)

Purpose: kill H1 before spending GPU hours on environment semantics. No Arcade involvement.

1. Launch a standalone probe script (default-OFF, no repo mutation): batch B ∈ {64, 256, 512, 1,024} particle states `[B, 8, 8, 8]`-shaped (or `[B, num_blocks, 8]`) through the shared frozen EFE scoring + egress head on the RTX 5090, with and without CUDA-graph capture of the rollout kernel.
2. Measure: steps/s, VRAM (MiB), achieved bandwidth fraction = `(B × 256 KiB) / (Δt × 2.1 TB/s)`, ESS under a synthetic score field, action entropy, kernel-launch count, launch overhead amortization.
3. Kill if: bandwidth fraction < 0.60 at B=1,024 with graphs, or steps/s ≤ arm-A-per-particle rate, or VRAM > 24 GiB at B=1,024.
4. Report `OBSERVED` numbers with the exact command, SHA, and GPU identity. Budget: ≤ 1 GPU-hour.

---

## 6. Resource limits

- **GPU:** single RTX 5090 (32 GB VRAM); no concurrent GPU jobs (exclusive scheduling).
- **VRAM:** B=1,024 states ≈ 256 MiB + hidden 2,048 × B (≈ 8 MiB) + logits + scratch ⇒ well under budget; enforce cap 24 GiB.
- **Host RAM:** B env instances (one per branch) — 1-by-1 streaming staging; cap concurrent live branches by host RAM; no batch grid materialization.
- **Disk:** remote ~97% full — artifacts written to `/tmp/p79f_phase8/` then pulled; worktree ops from inside the worktree only; scripts LF-normalized after scp.
- **Time:** probe ≤ 1 GPU-h; arm matrix ≈ 2–3 GPU-h per arm (sequential); total ≤ 1 shift.

---

## 7. Expected benefit and failure modes

- **Benefit:** if H2 holds, the first external scorecard progress in ARC-AGI-3; at minimum, a definitive falsification that candidate-count scaling alone (without new supervision) cannot produce progress — closing the "more compute" branch with evidence.
- **Failure modes:** (a) bandwidth undersaturation (K1); (b) particle collapse (K2); (c) zero progress (K3) — the likely outcome given E1/E2, in which case the verdict is `BLOCKED_NO_PROGRESS_EVENTS` and the next question is **supervision**, not compute; (d) infrastructure (K4); (e) **Coherent Solipsism** — internal free-energy minimization producing high internal coherence (r≈0.95) with zero external work; mitigated by the scorecard-only signal (E1/E2 precedent) and arXiv:2502.12118 (verifier-free test-time scaling is strictly suboptimal).
- **Mock-loop filters:** no simulated scorecards; no `reconstruct_scorecard`-style generators; a synthetic harness is not an unseen-environment benchmark; pytest passes are code-health only, never model-capability evidence.

---

## 8. Pre-registered gates (order enforced)

```text
probe H1 (K1/K2) → smoke (cn04, 3 rounds, rc=0, step-count arithmetic) → CUDA suite alone (candidate SHA, 368p baseline) 
→ discovery matrix (12 envs, arms A–D sequential) → aggregate env_count == 12, DONE only rc=0 
→ kill gate K3 (zero progress ⇒ BLOCKED_NO_PROGRESS_EVENTS: no SANS, no held-out, no promotion)
→ SANS gate (≥ 50 rows, ≥ 2 labels, ≥ 1 env, discovery-only, split-filtered) 
→ held-out split: sealed, never opened → evidence sealing (SHA-256 ledger, receipt) → separate promotion approval
```

---

## 9. Telemetry schema (compact, decision-relevant)

Per arm per env (immutable JSON): `env_id`, `split`, `arm`, `B`, `rounds`, `matched_branches`, `unmatched_branches`, `replay_mismatches`, `hidden_failures`, `steps`, `progress_events` (strict `levels_completed`), `unique_state_hashes`, `stationary_transition_rate`, `ess_min/mean`, `action_entropy`, `payload_diversity`, `k_psi` (descriptor), `sagnac_delta_mean`, `efe_decomposition`, `gpu_vram_mib`, `steps_per_s`, `bandwidth_fraction`, `launch_count`, `rc`, `candidate_sha`, `checkpoint_sha`, `overlay_sha`, `started_utc`, `finished_utc`, `schema_id`.

Aggregate: `env_count == expected_env_count` (12 per arm), `rc_sum == 0`, DONE marker only when all arms rc=0. Hash every artifact (SHA-256); receipt markdown; `diagnostic_only=true`.

---

## 10. Risks and open questions

1. Attained bandwidth < peak (arXiv:2605.30571): probe measures the real fraction before commitment.
2. Env stepping is sequential — the swarm parallelizes **planning**, not env simulation; total wall-clock may not improve even if GPU is saturated (report steps/s and env-step rate separately).
3. RESET-spam / stationary traps: eligibility-trace penalty (ν=−1 over horizon k=5) per corpus contract; stationarity veto Δ_grid==0 ⇒ Δ_Sagnac=1.0 ⇒ Q→−∞, +5.0 EFE penalty.
4. Post-outcome selection: prohibited (split frozen; top-K re-selection rejected at pre-registration).
5. SGLD adaptation of the action head remains blocked until the SANS gate opens (rows=0 sealed).

---

## 11. References

- Evidence: 7.9e receipt `829597c7…`; 7.9f receipt `4246443e…` (aggregate), artifacts `.../p79f_evidence/artifacts/p79f_discovery/`; Reference 3 contracts (`arc-curriculum-and-counterfactual-replay-contracts.md`).
- Code: `darwinian_phase_swarm.py`, `efe_planner.py`, `henri_decoder.py`, `arc_egress_contract.py`, `arc_action_payloads.py`, `arc_scorecard_delta.py`, `production_arc_run.py`, `arc_curriculum_replay.py` (all at base commit).
- Literature (OBSERVED via arXiv API): 2207.06649 (PMBS), 2404.16364 (ReZero), 2310.06513 (PTSA), 2410.23018 (PT for NQS), 2502.10328 (neural transports), 1811.04343 (Langevin PT), 2307.14804 (surprise-minimization collectives), 2502.12118 (verifier-free test-time scaling suboptimal), 2605.30571 (memory-bound decode).
- Corpus (INFERRED, bank `ca4bb787`): syncytium vs top-k MoE; chimera states / intermediate coupling; anisotropic Langevin; ε-spine 0.45–0.65; Coherent Solipsism (r≈0.95, loss −83.6%, ARC 0.0); exteroceptive Δν as the only honest valence.

---

## 12. Sign-off

This packet is pre-registered before any implementation. Any change to arms, budgets, thresholds, or the split requires a new pre-registration revision. Implementation begins only after: (1) this packet committed, (2) probe design approved, (3) one bounded default-OFF change at a time, (4) remote CUDA verification of each change.
