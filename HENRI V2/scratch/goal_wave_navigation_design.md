# Zone C State + Goal Wave + Latent Navigation
*2026-07-22 — Response to boundary axiom seeding & goal-conditioned planning questions*

---

## 1. Zone C TimescaleDB: What's Actually Stored (verified on 5090)

```
Table                           | Rows  | Has full waves? | Purpose
phylogenetic_engrams_65536      | 717   | YES (256 KB ea) | Past wave checkpoints + field channels
zone_c_resonant_hypersphere     | 6,937 | N/A (telemetry) | Per-step δ, coherence, phase records
zone_c_engrams                  | 366   | Partial          | Legacy stress-attributed engrams
```

**Breakdown by domain:**
| Domain | Engrams | Type |
|---|---|---|
| `arc3/ar25-0c556536` | 114 | Per-step wave checkpoints |
| `arc3/bp35-0a0ad940` | 92 | Per-step wave checkpoints |
| `arc3/cd82-fb555c5d` | 89 | Per-step wave checkpoints |
| `arc3/*/field_channel_consolidated` | 52 | L3 episode-end transition operators |
| `dc22-fdcac232_STEP_*` | 24 | Legacy format |

### Verdict: Seeded, but NOT functioning as boundary axioms

**What IS happening:** Every 5 steps, GRM retrieves the top-k most similar past engrams and blends them into the active wave (`0.7*live + 0.3*recalled`). This conditions the swarm relaxation — it biases which phase directions the wave moves during creep.

**What IS NOT happening:** The Zone C data does NOT enter the EFE scoring. The preference store (which DOES enter EFE via `beta_pragmatic * resonance`) remains **empty** on 0-score runs because `register_preference` only fires on WIN terminal.

**The data flow gap:**
```
Zone C (717 engrams) ──> GRM recall ──> conditions swarm relaxation ✓
                                          does NOT enter EFE scoring ✗
Preference store (0 engrams) ──> pragmatic_value resonance term ──> multiplies zero ✗
```

The Zone C is a **passive memory** — it stores what happened and blends it into the current state. But it doesn't tell the planner "this action is better than that action."

---

## 2. How to Establish an External Goal Wave

### The ARC task structure

ARC provides:
- **2-5 example pairs:** (input_grid_i, output_grid_i) — the system must infer the transformation
- **1 test input:** (test_input) — the system must produce the correct output

HENRI's current loop:
```
encode(test_input) → relax → plan(action) → step → observe → repeat
```
This is a REACTIVE loop — the system responds to what it sees, without knowing where it's going.

### What's missing: the goal wave

A goal wave is the VSA encoding of the DESIRED output grid. With it, the planner can score actions by: "does this action move me closer to the goal?"

The challenge for ARC: the goal wave is NOT KNOWN for the test input. It must be INFERRED from the examples.

### Three approaches (viability-ordered)

#### Approach A: Example-conditioned analogical inference (buildable now)

**Concept:** Encode each example input-output pair as a linked mapping in Zone C. When the test input arrives, retrieve the most similar example input, then apply its transformation pattern to generate the test output.

**Concrete implementation:**
```python
# Phase 1: Store examples (done once per ARC task)
for input_grid, output_grid in examples:
    input_wave = tokenizer.encode_spatial_grid(input_grid)
    output_wave = tokenizer.encode_spatial_grid(output_grid)
    # Store with domain tag linking input→output
    orch.checkpoint_wave(input_wave, domain=f"arc3/{task_id}/example_input")
    orch.checkpoint_wave(output_wave, domain=f"arc3/{task_id}/example_output",
                         linked_input_id=...)

# Phase 2: Infer goal for test input
test_wave = tokenizer.encode_spatial_grid(test_input)
# Retrieve most similar example INPUT wave
match = orch.segment_cache.retrieve(test_wave, domain_filter="example_input")
# Retrieve its paired OUTPUT wave as the goal
goal_wave = orch.segment_cache.retrieve_paired(match.id, domain_filter="example_output")

# Phase 3: Goal-conditioned planning
efe_with_goal = pragmatic_surprise - epistemic_gain
                + lambda_constraint * penalty
                + lambda_goal * sagnac_delta(predicted, goal_wave)
```

**What this needs (vs what exists):**
| Component | Status |
|---|---|
| VSA encode input/output grids | ✅ `o_vsa_ingress_tokenizer.encode_spatial_grid()` |
| Store linked pairs in Zone C | ⚠️ Store exists, linking doesn't |
| Retrieve most similar input | ✅ GRM retrieval exists |
| Retrieve paired output | ❌ Needs `linked_enngram_id` field + paired retrieval |
| Goal-distance term in EFE | ❌ Needs `goal_distance()` in `pragmatic_value()` |

**Effort:** ~80 lines across Zone C schema + EFEPlanner + production_arc_run.
**Risk:** The VSA encoding may not preserve enough spatial structure for reliable analogical matching. Test with known ARC tasks first.

#### Approach B: Learning the transformation function from examples

**Concept:** Train the transition model on example pairs: `transition(input_wave, learn_action) → output_wave`. Then apply the learned transformation to the test input.

**Implementation:**
```python
# Train on examples
for input_grid, output_grid in examples:
    input_wave = tokenizer.encode_spatial_grid(input_grid)
    output_wave = tokenizer.encode_spatial_grid(output_grid)
    # Learn: what action maps input → output?
    planner.train_transition_step(input_wave, special_learn_action, output_wave)

# Apply to test
test_wave = tokenizer.encode_spatial_grid(test_input)
goal_wave = planner.transition(test_wave, special_learn_action)
```

**Problem:** The transition model learns inter-frame dynamics (what happens when I press ACTION1), not task-level transformations (what is the rule?). The "special_learn_action" would need to encode the entire transformation — which is exactly what the system doesn't know.

**Verdict:** Not viable with current architecture. The transition operator maps (state, action) → next_state, not (input) → output. Different learning problem.

#### Approach C: Iterative refinement with environment feedback

**Concept:** Submit candidate outputs, get ARC verdict, refine.

**Implementation:**
```python
candidate_output = initial_guess(test_input)
for attempt in range(max_attempts):
    result = arcade.submit(candidate_output)
    if result == WIN:
        break
    # Use the failure signal to refine
    candidate_output = refine(candidate_output, result.feedback)
```

**Problem:** ARC gives binary feedback (WIN/LOSS) with no partial credit, no gradient. Refinement without gradient signal is random search.

**Verdict:** Only useful as a final verification step, not as a learning mechanism.

---

## 3. How to Navigate Latent Space: From Current State to Goal

Once we have a goal wave, the navigation problem becomes:

> Given current wave ψ_t and goal wave ψ_goal, find the sequence of actions [a_1, ..., a_k] that minimizes `sagnac_delta(ψ_{t+k}, ψ_goal)`.

### Current capability: one-step greedy

The EFE planner scores actions ONE STEP ahead:
```
EFE(a) = sagnac_delta(predicted_next, boundary) - beta * resonance + lambda * penalty
```

This is greedy — it picks the action that minimizes immediate surprise, not the action that leads to the goal.

### What's needed: multi-step planning

**Option 1: Goal-distance term in EFE (one-step greedy with goal bias)**
```python
def pragmatic_value_with_goal(predicted, boundary_axioms, goal_wave=None):
    surprise = min(sagnac_delta(predicted, axiom) for axiom in boundary_axioms)
    resonance = beta_pragmatic * max_resonance(predicted, preferences)
    goal_dist = sagnac_delta(predicted, goal_wave) if goal_wave is not None else 0.0
    return surprise - resonance + lambda_goal * goal_dist
```
This makes the planner prefer actions whose predicted outcomes are closer to the goal. Simple, one-line change.

**Option 2: Rollout planning (look-ahead with transition model)**
```python
def plan_toward_goal(state_wave, goal_wave, horizon=3, beam_width=4):
    beams = [(state_wave, [], 0.0)]  # (wave, action_seq, cumulative_goal_dist)
    for _ in range(horizon):
        new_beams = []
        for wave, actions, dist in beams:
            for action in candidate_actions:
                predicted = transition(wave, action_wave)
                new_dist = sagnac_delta(predicted, goal_wave)
                new_beams.append((predicted, actions + [action], new_dist))
        beams = sorted(new_beams, key=lambda x: x[2])[:beam_width]
    return beams[0][1]  # best action sequence
```
This uses the transition model to look ahead multiple steps. More expensive but handles non-greedy paths.

**Option 3: Gradient-guided navigation (continuous optimization)**
```python
# Treat the action as a continuous variable, optimize via gradient descent
action_wave = nn.Parameter(random_init)
optimizer = torch.optim.Adam([action_wave], lr=0.01)
for _ in range(100):
    predicted = transition(state_wave, action_wave)
    loss = sagnac_delta(predicted, goal_wave)
    loss.backward()
    optimizer.step()
best_action = snap_to_nearest_canonical_action(action_wave)
```
This optimizes a continuous action embedding toward the goal, then snaps to the nearest canonical action. Works when the action space is continuous.

### Which to build first?

| Option | Effort | Risk | When to use |
|---|---|---|---|
| Goal-distance EFE term | 5 lines | Low | Always — trivial to add, immediate effect |
| Rollout planning | ~40 lines | Medium | When greedy gets stuck in local minima |
| Gradient-guided | ~30 lines | Medium | When discrete actions are insufficient |

**Recommendation:** Start with Option 1 (goal-distance EFE term). It's 5 lines, zero risk, and gives the planner a reason to move toward the goal. The existing constraint channel and explore/exploit gate still function. The only change is: instead of minimizing surprise against the boundary axiom (prediction error), the planner ALSO minimizes distance to the goal.

---

## 4. The Immediate Actionable Items

### P0: Seed Zone C with example input-output pairs

Before HENRI can analogically infer goals, Zone C needs to store the EXAMPLE pairs from ARC tasks — not just past wave checkpoints.

**What exists:** Zone C stores per-step wave snapshots. These are "what the wave looked like at step N" — not "this input maps to this output."

**What to add:**
```sql
-- Store example pairs as linked engrams
-- example_input_wave and example_output_wave are both [num_blocks, 8]
-- linked_id ties them together
INSERT INTO phylogenetic_engrams_65536
  (id, timestamp, environmental_context_hash, semantic_index, engram_wave_bytes)
VALUES (gen_random_uuid(), now(), 'arc3/task_X/example_1_input', ...),
       (gen_random_uuid(), now(), 'arc3/task_X/example_1_output', ...);
```

### P1: Add `goal_distance` to `pragmatic_value`

In `efe_planner.py`, add an optional `goal_wave` parameter:
```python
def pragmatic_value(self, predicted_wave, boundary_axioms, goal_wave=None):
    # ... existing surprise + resonance ...
    goal_dist = 0.0
    if goal_wave is not None:
        g = goal_wave.view(-1)
        g = g / (torch.norm(g) + 1e-12)
        goal_dist = 1.0 - torch.dot(p, g)  # sagnac delta to goal
    return surprise - self.beta_pragmatic * resonance + self.lambda_goal * goal_dist
```

### P2: Analogical goal inference from Zone C

In `production_arc_run.py`, before the action loop:
```python
# Retrieve the most similar example input to the test input
test_wave = tokenizer.encode_spatial_grid(test_input)
match = orch.segment_cache.retrieve(test_wave, domain_filter="example_input")
if match.hits > 0:
    goal_wave = orch.segment_cache.retrieve_paired(match.top_id)
```

---

## 5. Why This Matters (connecting to the fallacy audit)

The nine fallacies identified earlier all trace back to ONE root cause: **the EFE landscape has no gradient toward task success.** Adding a goal wave creates that gradient.

| Fallacy | How goal wave fixes it |
|---|---|
| #1 Boundary axiom circularity | Goal distance replaces boundary axiom as primary EFE driver |
| #2 Empty preference store | Goal wave works immediately — no need for WIN to fill preferences |
| #3 Random action vectors | Planner selects actions that approach goal, not random directions |
| #4 EFE minimizes prediction error | EFE now minimizes GOAL DISTANCE, which IS task progress |
| #9 Self-referential objective | Goal wave is external — breaks the self-referential loop |

The goal wave is the single intervention that converts HENRI from a "physics simulator that scores 0.0" into a "goal-directed system that can solve tasks."
