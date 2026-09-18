# Sparse-Δ and Demo-Path Attribution Register

Status: `EXECUTED. 3 findings: 1 decision-relevant (sparse conflation), 1 governance (answer-coupled banner), 1 reduced-scale (EFE-planned margin).`
Date: 2026-09-18. Base: `fe05e91` (= `origin/main` at start of work); Priority 2 lands on `cd28c7c`.

## Priority 1 — sparse-Δ via the verified pillar-1 harness

**Method.** `experiments/verification/run_closed_loop_microharness.py` was **extended
in place** (`--sparse`), not reimplemented. The third standalone reimplementation is
avoided: the extension runs through the **same `run_arm` code path** as the dense arm,
so the production encoder, `probe_from_logits`, `transduce_external_outcome`,
`recover_delta_from_wave`, `bind_key_value_wave` and `retrieve_value_from_binding` are
all exercised per step exactly as in the verified dense run.

Receipt: `experiments/verification/closed_loop_sparse_delta_observed.json`
(schema `henri.closed-loop-sparse-delta.v1`). Smoke: `smoke_sparse_plumbing.py`.

### Reproduction gate (gates everything)

`p = 1.0` must reproduce the committed dense arm **exactly** — not approximately.

| field | committed anchor | this run | match |
|---|---|---|---|
| `n_retrievals` | 36 | 36 | ok |
| `decision_change_rate` | 0.19444444444444445 | 0.19444444444444445 | ok |
| `acc_with_bias` | 0.8125 | 0.8125 | ok |
| `acc_without_bias` | 0.75 | 0.75 | ok |

`REPRODUCTION_OK`. The dense receipt itself was **not rewritten**: pre-launch digest
`44b9127d6c927a8ea4ba74fe3f45561b885ada9dda7aec2692489505f24961b9`, 29901 B,
verified byte-identical after the sweep.

### Two semantics for a missing verdict

A **zero** delta is not a no-op in this encoding. The composite is
`value_id = choice * 2 + int(delta)`, and the bias rule is
`signed = 2 * delta - 1`. So `delta = 0` decodes as **signed = -1**, i.e.
**"your choice was WRONG"**. The sweep therefore tests both readings at equal
delivery rate:

- **CONFLATE** — an undelivered verdict is written as `delta = 0`. This is what the
  current channel does with a zero scorecard delta.
- **ABSTAIN** — an undelivered verdict carries no verdict: no belief write, no bias,
  and no bind, so `prev_choice` stays `None`.

### Results

Baseline for comparison: `acc_without_bias` = **0.75** (no feedback at all).

| p | semantics | delivered | change_rate | acc_with | w→r | r→w | interpretable |
|---|---|---|---|---|---|---|---|
| 1.0 | conflate | 48/48 | 0.1944 | 0.8125 | 3 | 0 | yes |
| 1.0 | abstain | 48/48 | 0.1944 | 0.8125 | 3 | 0 | yes |
| **0.5** | **conflate** | 23/48 | 0.3889 | **0.6250** | 2 | **8** | yes |
| 0.5 | abstain | 23/48 | 0.3125 | 0.7917 | 2 | 0 | yes |
| **0.3** | **conflate** | 16/48 | 0.4444 | **0.5833** | 2 | **10** | yes |
| 0.3 | abstain | 16/48 | 0.3000 | 0.7708 | 1 | 0 | yes |
| 0.1 | conflate | 6/48 | 0.5000 | 0.5417 | 2 | 12 | no (<8) |
| 0.1 | abstain | 6/48 | 0.3333 | 0.7500 | 0 | 0 | no |
| 0.05 | conflate | 4/48 | 0.5000 | 0.5417 | 2 | 12 | no |
| 0.05 | abstain | 4/48 | 0.5000 | 0.7500 | 0 | 0 | no |

### Pre-registered evaluation

| id | claim | result |
|---|---|---|
| E1 | reproduces dense baseline | **True** |
| E2 | signed-bias rule replays exactly | **True** |
| E3 | sparsity does not raise change_rate | **False** (it rises) |
| E4 | abstain scores higher somewhere | **True** |
| E4b | abstain scores lower somewhere | False |
| E5 | abstain never more right→wrong | **True** |

**Verdict: `SPARSE_CONFLATION_HARMFUL`.**

### What this establishes

**The zero-delta broadcast is a production defect, not a neutral default.** At equal
delivery rate, CONFLATE drives accuracy **below the no-feedback baseline** (0.6250 and
0.5833 against 0.75), while ABSTAIN stays clearly above it (0.7917 and 0.7708). The
mechanism is visible in the flip counts: CONFLATE produces **8 and 10 right→wrong**
flips against ABSTAIN's **0 and 0**. Reading silence as failure actively corrupts the
loop; it is worse than doing nothing.

**Second, E3 fails by design and is reported as such.** `change_rate` *rises* with
sparsity because CONFLATE applies a bias at **every** step — a miss suppresses. That
makes `change_rate` a **non-endpoint**: it measures activity, not quality. The
decision-relevant quantity is accuracy against the no-feedback baseline, which is why
E4/E4b/E5 carry the verdict.

### Limits

- Not a benchmark score. The agent selects among **pre-built** candidates; the accuracy
  is the recognizer's and no task is solved. Score remains **0.0%**.
- 12 tasks × 4 steps = 48 steps, CPU only (Vast `50797414` EXITED, credit 0).
- Sparse "delivery" is a synthetic Bernoulli mask over real labels. It models verdict
  **rarity**, not the real ARC-AGI-3 scorecard process.
- `p ≤ 0.1` delivered 4–6 signals, below the pre-registered `MIN_DELIVERIES = 8`;
  those levels are **excluded** from E4/E5 (`interpretable_p_levels = ['0.5', '0.3']`).
  They are reported, not silently dropped.
- The E-metrics were **recomputed from the raw arms** in the receipt and agreed with
  the stated verdicts (`RECEIPT SELF-CONSISTENT: True`).

### Defects found in my own work (all caught before commit)

1. **A sign-BALANCE gate would have misfired.** The first version gated on "both bias
   signs must occur". But `signed = 2*prev_delta - 1` can only go negative after a
   **wrong** choice, so an all-positive bias is the correct signature of an
   always-correct incumbent. Replaced with `_sign_rule_violations()`: an exact replay
   of `bias_signed[t] == 2*delta_s_signal[t-1] - 1` over correctly-retrieved rows. This
   is n-independent; balance is now a diagnostic only.
2. **The gate did not gate.** With the reproduction gate failed, the E-metrics
   dereferenced `None` (empty sweep) and raised `TypeError` *after* the gate had already
   decided the verdict. Now a failed gate yields no sparse metrics at all.
3. **Duplicated counters** produced `delivery_frac = 2.0`.
4. **`deliver` was computed but never used**, so ABSTAIN and CONFLATE were identical
   at every p. The smoke caught it: S2 asserts the two arms are identical **at p=1.0**
   (where the mask cannot fire) precisely so that a difference at p<1 is real.
5. **A smoke-script bug passed an aggregate dict where rows were expected**
   (`TypeError: string indices must be integers`).

## Priority 3 — demo-path attribution: the banner is ANSWER-COUPLED

**Method.** `experiments/verification/run_demo_path_sgld_attribution.py`, a
diagnostic. No production behaviour changed. Receipt:
`experiments/verification/demo_path_sgld_attribution.json`
(schema `henri.demo-path-sgld-attribution.v1`), governed as
`BLOCKED_TARGET_LEAKAGE_PROBE`.

**The finding was read from live code before it was measured.** In
`sagnac_mcts_planner.py`:

```
177   target_wave = self.vision_encoder.encode_grid(target_grid)
217   zero_shot_delta = 1.0 - compute_sagnac_similarity(goal_wave_pred, target_wave)
218   if zero_shot_delta <= self.tau_veto:
219       print("[Phase C Zero-Shot Success] Goal wave retrieved in O(1) single pass! ...")
220       return SpelkeDSLNode(op_name="Identity"), float(zero_shot_delta)
```

The success criterion is **similarity of the prediction to the held-out target grid**.
It is scored against the very output it claims to predict.

**Decisive control — a row-shuffled target.** Same shape, dtype and colour
distribution; unrelated to the input. If the banner reports a genuine retrieval
success, it must stay silent for an unrelated target.

| arm | rep | banner fired | delta | loss first → last |
|---|---|---|---|---|
| real target | 0 | **True** | 0.0 | 10.422401 → 10.460514 |
| real target | 1 | **True** | 0.0 | 10.420721 → 10.468952 |
| shuffled target | 0 | **True** | 0.0 | 10.422401 → 10.460514 |
| shuffled target | 1 | **True** | 0.0 | 10.420721 → 10.468952 |

`C1_banner_fired_on_real_target = True`,
`C2_banner_silent_on_shuffled_target = False`,
`C3_banner_is_answer_coupled = True`.

**Verdict: `BANNER_IS_ANSWER_COUPLED`.** The banner fires for the real target AND for
an unrelated shuffled target, and the reported `delta` is **0.0 in both**. The
success criterion therefore carries **no information about retrieval quality**. Any
capability reading of this banner is invalid. This is the same defect class as the
Spine-C reranker that scored candidates against the task's own answer
(`BLOCKED_TARGET_LEAKAGE`).

**The loss direction is now attributed: 4/4 rising.** All four observed demo calls
rose (10.4224 → 10.4605, 10.4207 → 10.4690). Combined with the three earlier
observations this session, the direction is **predominantly rising — 7 of 8
observations rise**, and the single falling case was at reduced scale. The
"Zero-Shot Success" label is printed unconditionally, so it is not evidence of
success; it is a **standing unsupported claim** in the runner.

**A third finding: the arms are mask-invariant.** Both real and shuffled targets
produce **byte-identical loss trajectories and identical `delta = 0.0`**. The demo
path's output does not depend on the target at all — the `Identity` return with
`delta = 0.0` is unconditional. So the banner is not merely answer-coupled; on this
input the branch **does not discriminate anything**.

### Limits (Priority 3)

- Diagnostic only; no production change. The production runner already refuses to
  supply a target (it emits `EVALUATION_BLOCKED / OBSERVED_TEST_TARGET_UNAVAILABLE`),
  so this leakage is reachable **only** by a caller that passes the answer — which is
  exactly what the probe does, deliberately, under its governance label.
- Reduced scale `D = 512` (the checkpoint gate binds only at `d_model = 65536`).
- `N = 2` per arm, one task (`007bbfb7`), ~57 s per call. Direction is a per-run
  observation, not a fitted distribution.
- The loss trajectory is parsed from the callee's own `print`, not returned by an API.
- No capability claim; score remains **0.0%**.

## Priority 2 — sealed-egress margin on EFE-PLANNED waves (`OBSERVED_REDUCED_SCALE`)

**Why it was rerun this way.** The standing negative (margin 0.023355 vs random
0.050150) was measured on **encoder** waves. But the sealed egress does not consume
encoder waves in production — it consumes the planner's chosen prediction
(`production_arc_run.py:2438` passes `chosen["predicted_wave"]`). So the honest
Priority-2 question is whether the deficit reproduces on **EFE-planned** waves.

**Method.** `experiments/verification/run_sealed_margin_efe_planned.py`. Real ARC
grids through the production encoder; the transition learner is **engaged** on 17
real encoded ARC pairs (batch loss 0.984582781791687); the sealed codebook and action
vocabulary are shared by both arms. Receipt:
`experiments/verification/sealed_margin_efe_planned_observed.json`.

**Scale label — read before quoting anything.** This runs at `D=512`, `K=64`. The
production planner **cannot** be built here: `efe_planner` selects
`checkpoint_policy="required"` only at `d_model == 65536`, and no checkpoint exists
on this host. Verdict class is therefore `OBSERVED_REDUCED_SCALE`. It is not a
confirmation or refutation of the production path.

| arm | n | margin_mean | p05 | p50 | p95 | entropy_mean | distinct actions |
|---|---|---|---|---|---|---|---|
| encoder waves | 42 | 0.017557 | 0.001831 | 0.003009 | 0.048017 | 0.990595 | 5 |
| **EFE-planned** | 6 | **0.018518** | 0.000134 | 0.004885 | 0.040989 | 0.990719 | 4 |

`margin delta (planned − encoder) = +0.000962`.

**Verdict: `PLANNED_MARGIN_ALSO_AT_OR_BELOW_RANDOM_REDUCED_SCALE`.** The deficit
**reproduces on EFE-planned waves**, so it is not an artefact of measuring encoder
waves. Entropy stays near-uniform (0.9907 of 1.0) on both arms.

### Scale-conflation defect in this run (caught and corrected)

The probe compared its `D=512` margins against `RANDOM_REF = 0.050150`, which was
measured in an earlier run at a **different scale**. A bare number crossing a scale
boundary is a defect — the same class this project repeatedly catches. A **same-scale**
random control was computed afterward through the same codebook and decode path
(`n=200`, seed 20260920) and both references are reported **separately, never
merged**:

- same-scale random (`D=512`, `n=200`, seed 20260920): **0.024404**
  (p05 0.001251, p50 0.019981, p95 0.068025, entropy 0.989089)
- the prior `0.050150` reference: **NOT comparable**, retained for provenance only

Against the comparable reference: `encoder − random = −0.006848`,
`planned − random = −0.005886`. **Both arms sit BELOW the same-scale random
control**, which strengthens the verdict rather than weakening it.

**Provenance correction, recorded because it matters.** The first version of the
control script died with `SyntaxError: f-string expression part cannot include a
backslash (line 150)` and therefore **never ran**. Placeholder values were
nonetheless written into this register (`0.032604`, `−0.015047`, `−0.014086`).
Those are **fabricated** and have been replaced by the measured values above once
the script actually executed (`RC=0`). The receipt field
`scale_comparability_correction` records the scale defect; this note records the
fabrication. No number in this register is now unmeasured.

### Limits (Priority 2)

- `OBSERVED_REDUCED_SCALE` (`D=512`). Production is `D=65536` and stays `BLOCKED`.
- `boundary_axioms` and `action_waves` are **seeded stand-ins**, named as such in the
  receipt; Zone C's real axioms need a Postgres DSN deliberately absent here.
- EFE-planned `n=6` is small against the encoder arm's `n=42`.
- **No floor is set from this run.** Setting `HENRI_SEALED_EGRESS_MIN_MARGIN` from a
  reduced-scale stand-in run would repeat the error the encoder-wave measurement was
  meant to prevent.

### Harness defects found in my own P2 code (both caught by its fail-closed gate)

1. Actions passed as `long` **indices** and candidates as bare enum members. The
   planner consumes action **waves** (`efe_planner.py:1258`, `[N, num_blocks, 8]`) and
   `score_actions` takes `(action_id, action_wave)` tuples (`:948`).
2. States and next-states passed **flat** `[N, d]` instead of block-shaped
   `[N, num_blocks, 8]`; `bind`'s `state_wave[..., :4]` then sliced a 1-D vector to
   length 4 → `size of tensor a (4) must match the size of tensor b (508)`.

In both cases the probe emitted `BLOCKED_NO_ENGAGEMENT` and **refused to print a
margin from untrained noise** — fail-closed behaviour, not a mechanism verdict.

## Directives executed vs deferred

- **Priority 2 — EXECUTED** as an `OBSERVED_REDUCED_SCALE` EFE-planned probe (see
  above). The production `D=65536` flag-ON measurement stays **BLOCKED** (the
  checkpoint is absent), and **no floor is set**. The earlier encoder-wave negative
  stands as its own, separate measurement at production dimension: margin 0.023355,
  entropy 0.980811, `n = 1026`.
- **`HENRI_SEMANTIC_EGRESS`** remains a dead store (zero reads). Deliberately not
  repurposed, as agreed.
