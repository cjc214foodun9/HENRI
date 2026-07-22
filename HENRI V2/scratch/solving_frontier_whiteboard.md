# Solving Frontier — HENRI Phase 2.8 Whiteboard
*2026-07-22 — Written in response to HANDOFF.md §9(D)*

---

## 1. The Core Problem (stated precisely)

**Scores: 0.0 despite full infrastructure working.** The constraint channel keeps predictions on-manifold (0% fallback). The swarm phase-locks (r≈0.95). The transition model learns (loss_ema tracks). The preference store fills. Zone C recalls. Everything WORKS — but nothing SOLVES.

### Why: the optimization landscape is task-agnostic

The EFE planner scores actions by:

```
EFE(a) = pragmatic_weight · pragmatic_surprise(a)
       − epistemic_weight · epistemic_gain(a)
       + λ · constraint_penalty(a)
```

Every term measures **internal dynamics coherence**, not task progress:

| Term | What it measures | Task-relevant? |
|---|---|---|
| `pragmatic_surprise` | Sagnac delta vs boundary axiom (prediction residual) | **No** — residual is `state − prev_predicted`, not `state − goal` |
| `epistemic_gain` | Retrieval entropy × latent novelty | **No** — novelty in 65536-dim latent space ≠ novelty in task space |
| `constraint_penalty` | Off-manifold residual of prediction | **No** — manifold is learned dynamics, not task constraints |

The only task-level signals that reach the system:
1. **WIN terminal** → valence=1.0 → preference consolidation → crystallized update
2. **RESET + no frame change** → valence=−1.0 → avoid null action
3. **Progress valence** → measures "change the learned physics admits," not "change that solves"

**The EFE landscape is flat with respect to task progress.** The planner cannot distinguish "action that moves toward solution" from "action that moves within learned dynamics" because both produce low surprise on a well-fit model.

---

## 2. Candidate Mechanisms (ordered by buildability)

### 2.1 Preference Resonance as Task Proxy (partially built)

**What exists:** `pragmatic_value()` includes `−β_pragmatic · max_resonance(predicted, prefs)`. This pulls the planner toward historically favorable outcome basins.

**What's missing:** The preference store is too sparse. At 3 envs × 30 steps, you get 0-3 entries (only WIN + positive valence trigger `register_preference`). Not enough density to steer.

**Fix path:** Seed preferences more aggressively. Option: register every frame transition that changes the grid state, not just WIN. The system doesn't know which changes are "good" — but MORE preferences give the resonance term more structure to work with. Risk: noise preferences dilute the signal.

### 2.2 PEARL Repair Loop (buildable now, ~20 lines)

**Concept:** When a candidate prediction is rejected (off-manifold), instead of dropping it, blend it toward the preference store to bring it back on-manifold. This creates preference-biased action selection.

**Why it addresses the gap:** The EFE landscape is task-agnostic, but the preference store encodes *some* task-relevant signal (waves from favorable transitions). PEARL repair forces off-manifold candidates toward the preference manifold — which happens to be the manifold of "transitions that led to good outcomes."

**Implementation:** See §3 below.

**Limitation:** Only helps if preferences contain task-relevant structure. With 0-3 entries, repair is a near-no-op. Pair with aggressive preference seeding.

### 2.3 Task-Conditioned Action Space (design required)

**Current state:** Actions are `ACTION1`–`ACTION4` with random-phase VSA encodings. The planner picks the one with lowest EFE — but all actions are semantically identical (random waves).

**What's needed:** Action waves that encode *specific grid transformations*. The VSA fractional binding already supports this — `move(Δx,Δy) ⊗ fill(color)` is a compositional action wave. But someone must define the primitives and map ARC game actions to them.

**Design question:** What are the ARC-relevant action primitives? The arcade environment supports: move (4 directions), select, fill, reset, submit. Can we encode these as structured VSA waves rather than random-phase vectors?

### 2.4 Goal-as-Attractor (requires goal specification)

**Concept:** If the planner had a "goal wave" (the desired output grid), EFE could minimize `sagnac_delta(predicted, goal_wave)` — actions that move toward the goal get lower EFE.

**Problem:** ARC doesn't provide a goal for the test input. You get input-output *examples* and must infer the transformation.

**Partial fix (JEPA-style):** Encode the example transformation pattern into the transition model. The model learns "input → output" dynamics from examples, then applies the learned transformation to the test input. This is what V-JEPA does — learn a predictor in representation space, then use it to generate the test output.

### 2.5 Transition Model Audit (research question)

**Question:** Does the learned transition model encode *anything* task-specific, or just generic smoothness/wave-propagation dynamics?

**Test:** After a run, compare the model's prediction for "fill action on example 1" vs the actual output of example 1. If the model learns task-specific dynamics:
- Predictions for task-relevant actions should match observed outcomes (low Sagnac delta)
- Predictions for task-irrelevant actions should not (high Sagnac delta)
- The EFE would then *implicitly* encode task progress (lower surprise = action consistent with learned task dynamics)

If the model only learns smoothness:
- All predictions have similar Sagnac delta
- The EFE landscape is truly flat with respect to task
- No amount of tuning β_pragmatic or λ will change scores

**This is the most important diagnostic.** It tells us whether the gap is in the scoring function (EFE needs task terms) or in the representation (the VSA encoding doesn't capture task structure).

---

## 3. PEARL Repair Loop — Concrete Design

### 3.1 What PEARL actually does (arXiv:2607.17917)

PEARL repairs noisy LLM-generated reasoning graphs by:
1. Materialize structural content from noisy output
2. Vote/filter into majority-correct anchors and rejected units
3. Judge-feedback local repair on rejected units
4. Strict dual-closure gate for acceptance
5. Typed residuals for non-accepted artifacts

### 3.2 HENRI analogical mapping

| PEARL concept | HENRI equivalent | Status |
|---|---|---|
| Materialize | `score_actions()` predicts candidate waves | ✅ Exists |
| Vote/filter | `constraint_penalty < reject_thresh` | ✅ Exists |
| Local repair | Preference-guided phase blend on rejected candidates | ❌ **TO BUILD** |
| Strict gate | `fallback_executed == false` | ✅ Exists |
| Typed residuals | `residual_type` field in telemetry | ❌ **TO BUILD** |

### 3.3 Repair algorithm

```
repair_candidate(predicted_wave, preference_store, α=0.3, max_attempts=3):
    """Attempt to bring an off-manifold prediction back on-manifold
    by blending with preference-store engrams."""
    
    penalty = constraint_penalty(predicted_wave)
    if penalty is None or penalty <= reject_thresh:
        return predicted_wave, "ACCEPTED_CLEAN", 0.0
    
    if preference_store.num_engrams() == 0:
        return predicted_wave, "REJECTED_NO_PREFS", penalty
    
    best_wave = predicted_wave
    best_penalty = penalty
    best_α = 0.0
    
    for attempt in range(max_attempts):
        # Retrieve top preference engram (or blend over top-k)
        sim = predicted_flat @ preference_store.engrams.T
        top_idx = sim.argmax()
        pref_wave = preference_store.engrams[top_idx]
        
        # Phase blend: steer toward historically favorable basin
        α_eff = α * (1.0 - attempt / max_attempts)  # reduce α on retry
        repaired = (1 - α_eff) * predicted_wave + α_eff * pref_wave
        repaired = repaired / (norm(repaired) + 1e-9)  # re-normalize
        
        new_penalty = constraint_penalty(repaired)
        if new_penalty is not None and new_penalty <= reject_thresh:
            return repaired, "ACCEPTED_PEARL_REPAIRED", new_penalty
        
        if new_penalty is not None and new_penalty < best_penalty:
            best_wave, best_penalty, best_α = repaired, new_penalty, α_eff
    
    # Couldn't fully repair; return best attempt
    return best_wave, "REJECTED_REPAIR_FAILED", best_penalty
```

### 3.4 Integration points

**In `EFEPlanner.score_actions()`** (after line 489):
```python
# PEARL repair: attempt to salvage rejected candidates
for r in results:
    if r["rejected"] and self.preference_store.num_engrams() > 0:
        repaired_wave, residual_type, new_penalty = self.pearl_repair(
            r["predicted_wave"])
        r["predicted_wave"] = repaired_wave
        r["constraint_penalty"] = new_penalty
        r["rejected"] = new_penalty > self.constraint_reject_thresh
        r["residual_type"] = residual_type
    else:
        r["residual_type"] = "ACCEPTED_CLEAN" if not r["rejected"] else "REJECTED_FALLBACK"
```

### 3.5 What this costs

- **Code:** ~30 lines in `EFEPlanner` (new method + integration)
- **Compute:** O(preference_store_size) per rejected candidate — negligible (<100 entries)
- **Risk:** Zero — repair only activates for rejected candidates; clean path unchanged
- **Test:** `test_pearl_repair` — construct off-manifold wave, verify repair moves penalty below threshold

---

## 4. β_pragmatic Tuning — Phase 2.8 Experiment Design

### 4.1 Hypothesis

Higher β_pragmatic increases the preference-resonance pull in EFE, which should:
- **Reduce RESET rate** (planner prefers actions whose predicted outcomes resemble historically favorable waves)
- **Increase action diversity** (preferences encode different favorable basins)
- **Potentially improve scores** IF preferences capture task-relevant patterns

### 4.2 ARM design

| ARM | β_pragmatic | Envs | Steps | λ_max | Reject threshold |
|---|---|---|---|---|---|
| ARM1 (baseline) | 1.0 | cd82, bp35, cn04 | 30 | 5.0 | 0.5 |
| ARM2 | 5.0 | cd82, bp35, cn04 | 30 | 5.0 | 0.5 |
| ARM3 | 10.0 | cd82, bp35, cn04 | 30 | 5.0 | 0.5 |

### 4.3 Pre-registered criteria

| Metric | Pass criteria | Fail criteria |
|---|---|---|
| EFE range | Bounded [−1.0, +2.0] | Explosion > +5.0 |
| Fallback rate | <5% steps | >20% steps |
| RESET rate | <20% | >40% |
| Preference fill | >0 entries per env | 0 entries all envs |
| Score | ANY non-zero | All 0.0 (baseline expected) |

---

## 5. Decision Tree

```
Transition model audit (§2.5)
├── Model learns task-specific dynamics?
│   ├── YES → Gap is in scoring. Add task terms to EFE.
│   │         • Goal-conditioned EFE (§2.4)
│   │         • Preference density increase (§2.1)
│   │         • PEARL repair for preference steering (§2.2)
│   └── NO  → Gap is in representation. VSA encoding doesn't capture task structure.
│             • Task-conditioned action space (§2.3)
│             • Input-output example encoding as conditioning
│             • Rethink: what does the VSA tokenizer actually encode?
└── PEARL repair
    ├── Preference store has entries?
    │   ├── YES → Repair should move rejected candidates toward favorable basins
    │   └── NO  → Seed preferences more aggressively first
    └── Repair reduces constraint_penalty for ≥1 candidate per step?
        ├── YES → Channel is functional; residual_type telemetry tracks it
        └── NO  → α too low, or preferences don't span the rejection region
```
