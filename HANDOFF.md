# HENRI — Project Handoff: Phase 3.3 — Learnable Actions + Calibrated Constraint
*Written 2026-07-22, end of Phase 3.3 build. Read this first.*

---

## Session Arc: What Was Built

Starting from Phase 3.1 (PEARL repair, scores 0.0), this session diagnosed and fixed three structural fallacies, calibrated the constraint boundary, and shipped learnable action embeddings.

### Phase 3.1: PEARL Repair Integration (commit `962e7a4`)
- `efe_planner.py`: Added `pearl_repair()` — preference-blend rejected candidates toward favorable basins
- `score_actions()`: Rejected candidates → PEARL repair → re-scored → can become admissible
- `residual_type` field: ACCEPTED_CLEAN / ACCEPTED_PEARL_REPAIRED / REJECTED_REPAIR_FAILED / REJECTED_NO_PREFS
- `production_arc_run.py`: `residual_type` in per-step telemetry
- Zombie PWM archived: `henri_pwm_orchestrator.py` → `_archive/orphans/`

### Phase 3.2: Preference-Blend Goal + HaPPY Audit (commit `80c8cb7`)
- `infer_goal_from_preferences()`: blends top-k preference engrams into a "desired outcome basin" goal wave
- `cross_cov_epistemic()`: singular value spectral entropy of [8×8] cross-covariance (not wired yet)
- 4-layer goal inference: Zone C → preference-blend → identity → None
- HaPPY paper (1503.06237) audited — key isomorphism documented: causal wedge → constraint channel, entanglement wedge → PEARL repair, erasure threshold → constraint_reject_thresh

### Phase 3.3: Learnable Action Wave Embeddings (commit `a3eeef6`) — **CURRENT HEAD**
- **Fallacy #3 fix:** `action_embeddings: nn.Parameter [num_actions, num_blocks, 8]` in `EFEPlanner`
- Trained alongside transition model via Sagnac loss gradients (`action_lr_scale=0.2`)
- Per-block re-normalized after each update
- `LEARNABLE_ACTIONS` env flag (default OFF per staged-change convention)
- `GRID_DIST_EPISTEMIC` env flag defined but not wired (P2)

---

## Benchmark History

### Benchmark 1: Over-Constrained Paralysis (thresh=0.25, PV=0)
| Metric | Value |
|---|---|
| Steps | 527 |
| EFE mean | +1.870 |
| Rejection | 89.9% |
| Preference max | 0 |
| WINS | 0 |

**Diagnosis:** Threshold at 0.25 sits inside the physical manifold's noise floor (natural residuals 0.30-0.35). 53-94% candidates rejected → planner blind.

### Benchmark 2: Calibrated (thresh=0.38, PV=1) — `production_run_1784749966`
| Metric | Value |
|---|---|
| Steps | 495 |
| EFE mean | **+1.028** (↓45% from B1) |
| EFE min | **−1.763** |
| Rejection | **0.0%** |
| Preference max | **179** |
| WINS | 0 |

**Key:** 0% rejection, 179 preferences accumulated, preference-blend goal active on 9/10 envs, Zone C analogical on 1 env. First environment-wide negative EFE (g50t: −0.27).

### Benchmark 3: Learnable Actions (LEARNABLE_ACTIONS=1) — 🔄 RUNNING
- Log: `bench_20260722_211246_learnable.log`
- Telemetry: `production_run_1784_______` (TBD)
- 10 envs, 50 steps, thresh=0.38, PV=1, λ=1.0, β=10.0
- **Last check:** 447 steps, env 9/10, 0 WINS, EFE negative on ar25

---

## Current Codebase State

| File | Lines | Key changes |
|---|---|---|
| `efe_planner.py` | 995 | pearl_repair, infer_goal_from_preferences, cross_cov_epistemic, learnable action_embeddings, constraint_reject_thresh=0.38 default |
| `darwinian_phase_swarm.py` | 509 | learnable_actions threaded to planner, candidate_action_waves conditional |
| `production_arc_run.py` | 604 | LEARNABLE_ACTIONS, GRID_DIST_EPISTEMIC, PROGRESS_VALENCE, 4-layer goal inference |
| `hopfield_cleanup.py` | 188 | Unchanged |
| `functor_flow.py` | 433 | Unchanged |

### Env Tunables (all)
| Var | Default | Effect |
|---|---|---|
| `CONSTRAINT_AXIOM` | 0 | Arm constraint scalars |
| `PROGRESS_VALENCE` | 0 | Seeds preference store from within-invariant motion |
| `LAMBDA_CONSTRAINT_MAX` | 5.0 | Exactness cap on accuracy-gated λ |
| `CONSTRAINT_REJECT_THRESH` | **0.38** | Hard-rejection cutoff (calibrated from telemetry) |
| `BETA_PRAGMATIC` | 1.0 | Preference-resonance strength (10.0 proven effective) |
| `LAMBDA_GOAL` | 0.0 | Goal-distance weight (1.0 proven effective) |
| `LEARNABLE_ACTIONS` | 0 | Learnable action wave embeddings (NEW) |
| `GRID_DIST_EPISTEMIC` | 0 | Pixel-wise grid delta as epistemic signal (NEW, not wired) |

---

## Remaining Fallacies (from original 9)

| # | Fallacy | Status | Fix |
|---|---|---|---|
| 1 | Boundary axiom circularity | ⚠️ Partial | Goal wave breaks loop but needs meaningful target |
| 2 | Preference store dead | ✅ Fixed | β=10.0 + PROGRESS_VALENCE fills 179 entries |
| 3 | Actions as random vectors | ✅ Fixed | Learnable action embeddings (`a3eeef6`) |
| 4 | EFE minimizes prediction error | ⚠️ Partial | Goal wave + preference resonance added |
| 5 | Transition model trains on wrong objective | ⚠️ Partial | Learnable actions help but inter-frame ≠ task transform |
| 6 | Novelty in latent space ≠ task relevance | 🔴 Open | GRID_DIST_EPISTEMIC defined but not wired |
| 7 | Zombie PWM orchestrator | ✅ Fixed | Archived to `_archive/orphans/` |
| 8 | GPU verification = physics-only | 🔴 Open | No task-level tests |
| 9 | Root tautology | ⚠️ Partial | Preference engine breaks self-reference |

---

## Next Session Priorities

### P0: Pull & Analyze Phase 3.3 Benchmark
- Telemetry at `production_run_1784_______` on 5090
- Compare action embedding divergence vs random-phase baseline
- Check if learnable actions change action distribution or EFE

### P0: Wire GRID_DIST_EPISTEMIC (Fallacy #6)
- Compute pixel-wise frame delta in `production_arc_run.py`
- Thread as epistemic multiplier to planner
- ~15 lines, zero risk

### P1: Action Embedding Divergence Metric
- Track cosine distance between action embeddings over time
- If embeddings diverge → model is learning action-specific effects
- If embeddings stay correlated → learnable actions not helping

### P1: Task-Conditioned Action Initialization
- Initialize learnable embeddings from fractional binding of spatial primitives
- Gives the model a structured prior instead of random init
- `action_embeddings[i] = tokenizer.encode_spatial_grid(primitive_pattern[i])`

### P2: Zone C Example Pair Seeding
- Store linked (input, output) pairs in Zone C for analogical goal inference
- Currently Zone C retrieval triggered on 1/10 envs (top_sim=0.702)

---

## Infrastructure

| Resource | Location |
|---|---|
| **Repo** | `github.com/cjc214foodun9/HENRI`, branch `main`, HEAD `a3eeef6` |
| **5090** | `ssh -p 53976 root@62.107.25.198`, repo `/workspace/HENRI/HENRI V2` |
| **CI cronjob** | `8027351ab01e` — delivery broken (Photon sidecar), script executes fine |
| **Suite CPU** | 51 passed, 1 skipped |
| **Suite CUDA** | 52 passed (verified on 5090 at `8cbf644`) |
| **NotebookLM** | bank `ca4bb787-de9d-4ee0-89c9-bf71259cc86d` |
| **Zone C** | 717 engrams across envs |
| **Skills** | henri-soul, henri-architecture, arxiv, notebooklm |
| **Handoff** | This file (`HANDOFF.md`) |

### Manual 5090 Recovery (CI delivery broken)
```bash
# Sync
ssh -p 53976 root@62.107.25.198 "cd '/workspace/HENRI/HENRI V2' && git reset --hard origin/main"
# Run CUDA suite
ssh ... "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m pytest tests/test_henri_core.py -q"
# Launch experiment
ssh ... "cd '/workspace/HENRI/HENRI V2' && setsid bash -c 'nohup bash phaseXY.sh > logs/phaseXY.log 2>&1 &'"
# Pull telemetry
scp -P 53976 root@62.107.25.198:/workspace/HENRI/HENRI\ V2/telemetry_logs/production_run_*.jsonl .
```

### Benchmark Recovery
```bash
# Phase 3.3 benchmark running on 5090. To check:
ssh -p 53976 root@62.107.25.198 'tail -n 5 /workspace/HENRI/HENRI\ V2/logs/bench_20260722_211246_learnable.log'
# When complete, pull:
scp -P 53976 "root@62.107.25.198:/workspace/HENRI/HENRI\ V2/telemetry_logs/production_run_1784*" .
```