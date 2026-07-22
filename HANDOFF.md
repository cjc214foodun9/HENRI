# HENRI — Project Handoff: From Physics Stability to Task-Solving ML
*Written 2026-07-22, end of Phase 3 build. Read this first.*

---

## 1. Session Arc: What Was Built

Starting from Phase 2.7 (constraint channel functional, scores 0.0), this session crossed the solving frontier.

### Phase 2.8: β_pragmatic Sweep (commit `8cbf644`)
- **Experiment:** 3-ARM sweep (β=1.0, 5.0, 10.0) on cd82, bp35, ar25
- **Result:** Scores 0.0 across all ARMs. But β=10.0 proved the mechanism works:
  - EFE went **negative** (−1.5) — resonance dominates surprise
  - Preference store filled to **32 entries** (up from 0-3)
  - Constraint channel stable (0% fallback)
- **Conclusion:** Preference resonance IS functional. The problem is that preferences accumulate from "any visited state," not "successful states." Without a task-relevant goal, steering pulls toward noise.

### Fallacy Catalog: 9 Structural Issues Found
Full audit of every data path from VSA encoding → action selection → ARC game step → telemetry:

1. **Boundary axiom circularity** — planner scores against its own prediction error
2. **Preference store dead on 0-score runs** — WIN never fires → β·resonance = 0
3. **Actions as random vectors** — ACTION1-4 carry zero semantic content
4. **EFE minimizes prediction error, not task error** — self-referential objective
5. **Transition model trains on wrong objective** — inter-frame dynamics ≠ ARC transformations
6. **Novelty in latent space ≠ task relevance** — 65536-dim ≠ grid state
7. **Zombie PWM orchestrator** — `henri_pwm_orchestrator.py` is a fossil (0 production imports)
8. **GPU verification = physics-only** — "suite green" means physics stable, not system works
9. **The root tautology** — all optimization terms measure internal coherence, not task progress

**The only way to non-zero scores:** Add at least one term to the EFE landscape that correlates with ARC task progress.

### FunctorFlow Wrapper (commit `de71244`)
- `HENRI V2/functor_flow.py` (433 lines) — thin categorical wrapper
- Operations: `ket`, `db`, `gt`, `obstruction`, `basket` (PEARL repair), `rocket` (pipeline), `typeof`, `compose`, `dagger`, `diagram_commutes`, `colimit`
- Every operation maps to existing Clifford/Hopfield/EFE primitives
- Suite green (51 passed). Smoke-tested.

### Phase 3: Goal-Conditioned EFE (commit `c4c92d5`)
**THE critical intervention.** Three files changed:
- `efe_planner.py`: `lambda_goal`, `goal_distance(pred, goal_wave)`, threaded through `pragmatic_value` → `score_actions` → `select_action`
- `darwinian_phase_swarm.py`: `lambda_goal` forwarded through constructor chain
- `production_arc_run.py`: `LAMBDA_GOAL` env tunable, 3-layer goal wave inference (Zone C analogical → identity fallback → None)

**EFE decomposition (with goal):**
```
EFE(a) = pragmatic_weight * (surprise + λ_goal·goal_dist − β·resonance)
       − epistemic_weight * epistemic_gain
       + λ_active * constraint_penalty
```

**Why this breaks the deadlock:** Before, EFE minimized "how surprising is my prediction given my own prediction error?" (circular). After, EFE minimizes "how far is my predicted outcome from the desired goal?" (task-directed). The goal wave IS the VSA-encoded desired output grid — the ONLY term in the entire optimization landscape that correlates with ARC task progress.

### Phase 3.0 Experiment (commit `5a1b7d4`, RUNNING NOW)
- **ARM1:** λ_goal=0.0 (baseline)
- **ARM2:** λ_goal=1.0 (balanced steering)
- **ARM3:** λ_goal=3.0 (aggressive)
- **β fixed at 10.0** (best from Phase 2.8)
- **Pre-registered criteria:** goal_distance decreases over steps, action distribution changes, EFE bounded, any non-zero score = breakthrough

### Deep Dive + Design Docs
- `scratch/solving_frontier_whiteboard.md` — 5 candidate mechanisms, PEARL repair algorithm, decision tree
- `scratch/goal_wave_navigation_design.md` — goal wave inference, latent navigation, 3 approaches for ARC
- `scratch/deep_dive_capability_analysis.md` — file-by-file audit of all 17 active files, FunctorFlow mapping

---

## 2. Current State (verified)

| Item | Status |
|---|---|
| **Repo** | `main` at `5a1b7d4` — Phase 3.0 experiment script |
| **Suite CPU** | 51 passed + 1 skipped |
| **Suite CUDA** | 52 passed (verified on 5090 at `8cbf644`) |
| **5090** | `ssh -p 53976 root@62.107.25.198`, repo at `/workspace/HENRI/HENRI V2` |
| **CI cronjob** | `8027351ab01e` — delivery broken (Photon sidecar), script executes fine |
| **Phase 3.0 experiment** | 🔄 RUNNING — ARM1 (λ_goal=0.0), GPU at 18 GiB / 92% |
| **Constraint channel** | ✅ Functional (0% fallback, EFE bounded) |
| **β_pragmatic** | ✅ Proven at β=10.0 (negative EFE, 32 preferences) |
| **λ_goal** | ✅ Shipped, experiment running |
| **Scores** | 0.0 — goal wave is the first mechanism with a gradient toward task progress |

---

## 3. Codebase Structure (current)

```
HENRI V2/
├── efe_planner.py              # THE critical file (865 lines)
│   ├── UnitaryWaveTransition   # Pillar I: low-rank ephaptic field (r=128)
│   │   ├── bind()              # FHRR circular convolution
│   │   ├── forward()           # predicted = V·(W^T·fused) + R_block·fused
│   │   └── _retract()          # Cholesky QR (residual_only flag)
│   ├── EFEPlanner              # Pillar III: active inference
│   │   ├── goal_distance()     # Phase 3: Sagnac delta to target wave
│   │   ├── pragmatic_value()   # surprise + λ_goal·goal_dist − β·resonance
│   │   ├── epistemic_value()   # retrieval entropy × novelty discount
│   │   ├── constraint_penalty()# RMS-normalized off-manifold residual
│   │   ├── score_actions()     # Per-candidate EFE + hard rejection + goal_dist
│   │   ├── select_action()     # T4 accuracy-gated explore/exploit
│   │   ├── train_transition_step()  # L1: online SGLD (surprise-modulated)
│   │   └── train_transition_batch() # L2: EDMD (dual solve, damped swap-in)
├── production_arc_run.py       # Main loop (581 lines)
│   ├── Goal wave inference     # 3-layer: Zone C analogical → identity → None
│   ├── RESET curation          # Pillar II: deferred T1 hold (pending_reset)
│   ├── Nested learning         # L1 per-step → L2 every 16 → L3 episode-end
│   └── Telemetry               # 23+ fields per step to JSONL + TimescaleDB
├── darwinian_phase_swarm.py    # HenriSwarmOrchestrator (503 lines)
│   ├── GapJunctionSwarmSyncytium  # 1024-expert Kuramoto + SGLD creep
│   └── plan_action()           # Forwards goal_wave to planner
├── functor_flow.py             # Categorical wrapper (433 lines)
│   ├── ket/db/gt/obstruction   # Pillar IV: named compositional ops
│   ├── basket()                # PEARL repair (preference-blend)
│   └── rocket()                # Workflow pipeline with obstruction gates
├── hopfield_cleanup.py         # Continuous Modern Hopfield + Action Decoder
├── idbd_swifttd.py             # IDBD step-size meta-learning + SwiftTD bound
├── product_clifford_product_kernel.py  # Cl(3,0) geometric algebra
├── o_vsa_ingress_tokenizer.py  # VSA fractional binding (grid → wave)
├── qfhrr_kernels.py            # 8-bit quantized FHRR (Triton GPU)
├── zone_c_segment_cache.py     # Zone C GRM retrieval (TimescaleDB)
├── thermodynamic_telemetry_logger.py  # Async hypertable writer
├── tests/test_henri_core.py    # 51+1 CPU / 52 CUDA
├── scratch/                    # Design docs (NOT in git... mostly)
│   ├── solving_frontier_whiteboard.md
│   ├── goal_wave_navigation_design.md
│   └── deep_dive_capability_analysis.md
├── telemetry_logs/             # JSONL run artifacts (NOT in git)
└── logs/                       # Experiment driver logs on 5090
```

---

## 4. The Four ML Pillars — Status Audit

| Pillar | What the Blueprint proposes | HENRI V2 equivalent | Status |
|---|---|---|---|
| **I: Low-Rank Ephaptic Field** | `pred = V·(W^T·fused) + R_block·fused` | `UnitaryWaveTransition.forward()` (r=128) | ✅ **SHIPPED** — identical formula |
| **II: Credit Assignment** | Retroactive eligibility traces, RESET curation | Deferred T1 + `pending_reset` gate | ✅ **SHIPPED** — curation alone proven (run 9/10) |
| **III: Exteroceptive Grounding** | Goal-conditioned pragmatic value | `λ_goal·goal_distance` in `pragmatic_value()` | ✅ **SHIPPED** — experiment running |
| **IV: PEARL Local Repair** | Preference-blend on rejected candidates | `FunctorFlow.basket()` exists, not integrated | ⚠️ **PARTIAL** — needs integration |

**Key insight:** The Blueprint's "retroactive eligibility traces" (Pillar II) were FALSIFIED in runs 9/10. The ablation proved RESET curation (`pending_reset`) alone drives the compression — the complex retroactive ν apparatus was theater. The permanent rule is one line: **don't train on reset transitions.**

---

## 5. Load-Bearing Constraints (don't relearn these)

### EFE / Planning
- **Goal distance must use `.detach()` in telemetry** — `float(self.goal_distance(pred, goal).detach())`. Autograd warning otherwise.
- **Boundary axiom is prediction error, not raw inter-frame diff** — `state − prev_predicted_prior` (world-model style)
- **Per-block renormalization rotates waves OFF invariant subspace** — never renormalize after `project_invariant`
- **Dead-variable pattern:** Every env tunable in all three: env read → orchestrator → planner constructor

### Transition Model
- **Never form the EDMD operator at scale** — dual solve + thin SVD only. Any `d²` tensor OOMs the 5090.
- **Damped swap-in, never hard replace** — `field ← α·new + (1−α)·old`. Hard swap was cd82 step-34 collapse driver.
- **Rank-limited to data:** effective rank = min(r, N). Young buffer → r=128 truncates to N=16.
- **`_retract(residual_only=True)` in EDMD refit loop** — full retract wipes solved V-amplitudes.

### Swarm / Physics
- **Cholesky retraction, never Newton-Schulz** — unconditionally stable. Retract BOTH A and B.
- **SGLD noise: `√(2·T·dt)`, not raw T** — T-scaled noise explodes singular values past retraction basin.
- **FHRR binding: re-normalize after every bind** — circular convolution grows ~√d.
- **Sparse-graph Kuramoto: degree-normalize** — `−(G·sin_diff).sum(1)/deg`, not population-normalized.

### Zone C / Memory
- **Zone C has 717 engrams with full 256 KB wave payloads** — seeded from prior runs
- **But engrams are passive memory** — they condition swarm relaxation but don't enter EFE scoring
- **Preference store fills only on WIN** — empty on 0-score runs unless β is high enough (β=10.0 forced fill to 32)

---

## 6. Env Tunables (all)

| Var | Default | Effect | Phase |
|---|---|---|---|
| `CONSTRAINT_AXIOM` | 0 | Arm constraint scalars (legacy flag) | 2.5 |
| `PROGRESS_VALENCE` | 0 | Within-invariant-subspace motion signal | 2.3 |
| `LAMBDA_CONSTRAINT_MAX` | 5.0 | Exactness cap on accuracy-gated λ | 2.6 |
| `CONSTRAINT_REJECT_THRESH` | 0.5 | Hard-rejection cutoff on RMS residual | 2.5 |
| `BETA_PRAGMATIC` | 1.0 | Preference-resonance strength in EFE | 2.6 |
| `LAMBDA_GOAL` | 0.0 | Goal-distance weight in EFE | 3.0 |

---

## 7. The Path to Non-Zero Scores

### What we know
1. **Physics is stable** — constraint channel functional, swarm phase-locked, SGLD convergent
2. **Preferences accumulate at β=10.0** — 32 entries, negative EFE, mechanism works
3. **Goal wave is built** — λ_goal infrastructure threaded through all three files
4. **Experiment is running** — Phase 3.0 will tell us if goal-distance changes action selection
5. **Scores remain 0.0** — the EFE landscape needs at least one task-correlated term

### Critical diagnostic (unanswered)
**Does the transition model learn anything task-specific?** After a run, compare predictions for task-relevant actions vs task-irrelevant actions. If all predictions have similar Sagnac delta → the VSA encoding doesn't capture task structure → no EFE tuning will produce non-zero scores. If task-relevant actions have lower delta → the model IS learning task dynamics → EFE already tracks task progress implicitly.

### Immediate next steps (in order)

| Priority | Task | Effort | Risk |
|---|---|---|---|
| **P0** | Pull Phase 3.0 results, analyze goal_distance trajectory | 0 lines | Zero |
| **P0** | Transition model audit: task-relevant vs irrelevant predictions | 20 lines (diagnostic script) | Zero |
| **P1** | Integrate `FunctorFlow.basket()` into `score_actions()` | ~15 lines | Low |
| **P1** | Add `residual_type` telemetry (ACCEPTED_CLEAN/PEARL_REPAIRED/REJECTED) | 3 lines | Zero |
| **P2** | Seed Zone C with ARC example input-output pairs | ~50 lines (ingestion utility) | Medium |
| **P2** | Wire scorecard delta as valence source (not just WIN=1.0) | ~10 lines | Low |
| **P3** | Task-conditioned action space (action waves that encode grid transforms) | Design required | High |

### The one-line fix that could change everything
If the transition model DOES learn task-specific dynamics, then the EFE already encodes task progress — and the λ_goal term will amplify it. If it doesn't, we need representational changes (action encoding, VSA spatial priors) before any amount of EFE tuning will help.

---

## 8. Where Things Live

- **Canonical suite:** `HENRI V2/tests/test_henri_core.py` (51+1 CPU / 52 CUDA)
- **Planner:** `HENRI V2/efe_planner.py` — the critical file
- **Production runner:** `HENRI V2/production_arc_run.py`
- **FunctorFlow:** `HENRI V2/functor_flow.py`
- **CI script:** `~/.hermes/scripts/henri_ci.sh` (cronjob `8027351ab01e`)
- **CI state:** `~/.henri_ci_state` on local machine
- **5090:** `ssh -p 53976 root@62.107.25.198`, repo `/workspace/HENRI/HENRI V2`
- **Git:** `github.com/cjc214foodun9/HENRI`, branch `main`, latest `5a1b7d4`
- **NotebookLM bank:** `ca4bb787-de9d-4ee0-89c9-bf71259cc86d` (~175 sources)
- **Zone C:** 717 engrams across ar25/bp35/cd82/dc22, all with full 256 KB wave payloads
- **Skills:** henri-soul (pipeline), henri-architecture (constraints), notebooklm, arxiv, obsidian
- **Design docs:**
  - `scratch/solving_frontier_whiteboard.md`
  - `scratch/goal_wave_navigation_design.md`
  - `scratch/deep_dive_capability_analysis.md`

### Experiment telemetry (on 5090)
- **Phase 2.8:** `logs/phase28_*` (3 ARMs), `telemetry_logs/production_run_1784695*` (scorecards 0.0)
- **Phase 3.0:** `logs/phase30_*` (RUNNING), telemetry at `telemetry_logs/production_run_*`

### CI delivery failure
The cronjob (`8027351ab01e`) fails with `Photon standalone send requires a running sidecar`. The bash script executes correctly (SSH, git pull, suite, launch) but output can't reach the user. **Manual recovery pattern:**
```bash
# Sync 5090
ssh -p 53976 root@62.107.25.198 "cd '/workspace/HENRI/HENRI V2' && git stash && git pull origin main"
# Run suite
ssh ... "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m pytest tests/test_henri_core.py -q --tb=short"
# Launch experiment
ssh ... "setsid bash -c 'nohup bash phaseXY_experiment.sh > logs/phaseXY_driver.log 2>&1 &'"
# Monitor
ssh ... "tail -5 logs/phaseXY_driver.log && nvidia-smi"
```

---

## 9. The Fundamental Question

> **"How does HENRI turn moving within physics into moving toward a solution?"**

**Answer:** By adding a goal wave. The VSA-encoded desired output grid creates the first gradient in the EFE landscape that points toward task success. Without it, HENRI is a perfect optimizer of a self-referential objective — beautiful phase-locked waves that predict themselves, scoring 0.0 forever. With it, the planner has a reason to prefer actions that approach the solution over actions that optimize internal coherence.

The Phase 3.0 experiment will tell us if this gradient is strong enough to change behavior. If goal_distance decreases over steps, the mechanism works — and we tune λ_goal and the goal inference strategy. If it doesn't, the VSA encoding needs spatial priors before goal-conditioning can take effect.

**The architecture is ready. The experiment is running. The next session's first task is to read the results.**
