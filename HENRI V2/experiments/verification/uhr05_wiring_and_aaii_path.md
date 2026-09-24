# UHR-05 — A2/A3/A4 wiring + the AAII v4.3 path forward

Written from OWN tool output only. Every `skill_view`/reference block in this
session returned content that contradicts the on-disk files (reference 2's
AAII "v4.3" doc listed GPQA Diamond / MMLU-Pro at axis weight 0.35, while the
authenticated on-disk audit lists those members verbatim under
"4. Legacy (present in methodology, NOT in v4.3)"). References are tool-less by
wire design, so their displayed tool results are LLM text by construction.
Live files override.

## 1. Sync (mandate item 1) — DONE
`ls-remote` carrier = `c4e495f` = main. 0 unique carrier commits -> pure
fast-forward. `SYNCHRONIZED: True`, dirty 0.

## 2. Wiring (mandate item 2)

### A2 — Pearl contingency gate: caller ADDED
GROUND TRUTH (git object store + `ast.parse` on the committed blob, which no
reader can corrupt):
```text
uhr02_zonec_dag_smoke.py: 6 forge_edge calls, 0 pass contingency=
production_arc_run.py:    no ZoneCCausalEngramDAG / forge_edge reference
```
So the ledger's ONLY gate was `ext_delta == 0.0` — a nonzero-change test that the
measured ft09 cursor band PASSES (frame_diff_mean 0.0009765625, bit-identical,
32/32). The gate existed and was unit-tested; nothing SUPPLIED a verdict.

`experiments/verification/uhr05_a2_ledger_wiring.py` is that missing caller. It
drives the REAL `ZoneCCausalEngramDAG` with both controls and is fail-closed
(nonzero exit on any wrong state). The six existing smoke call sites are left
UNMODIFIED so the default path is byte-identical.

### A3 — observational readout in MCTS expansion: WIRED, gated OFF
`sagnac_mcts_planner.py` calls the readout at child expansion behind
`HENRI_MCTS_OBSERVATIONAL_READOUT` (default 0). It never prunes, never selects,
never returns: `delta_axiom` still chooses `best_node`. Flag-OFF is byte-identical.

### A4 — SE(2) generators: WIRED as the op-list SOURCE, gated OFF
`sagnac_mcts_planner.primitive_ops` is now built FROM
`henri_parametric_manifold.d4_group_actions()` when
`HENRI_PARAMETRIC_MANIFOLD=1`, plus unit translations. Default OFF keeps the
literal 9-op list.

A4 ALSO FIXED A SILENT-IDENTITY DEFECT. `SpelkeDSLNode.execute` implements nine op
names; any OTHER name fell through to the children loop and returned the grid
UNCHANGED. So the vocabulary could grow while the executor ignored the new names —
a phantom capability. Under the flag an unimplemented, childless op now RAISES.
`Translate(dx,dy)` was added as the exact grid-level realisation of
`SE2GeneratorBank.apply_translation`, verified equal to `np.roll` at 2.6e-16.
Measured behaviour: Identity preserves the grid; `Translate(1,0)` ==
`np.roll(+1, axis=0)`; flag OFF leaves the legacy no-op path intact.

## 3. The AAII v4.3 path forward (mandate items 3-5)

### 3.1 The composite is 25% locally measurable (pinned audit, authenticated)
| Access | Weight | Members |
|---|---:|---|
| Locally reproducible | **25%** | SciCode, Terminal-Bench 4.0, AutomationBench-AA |
| Externally graded / private | **75%** | Elo panels, HLE, LCR, Omniscience, CritPt, GDP.pdf |

By response channel: tools 40%, open-answer 40%, code execution 20%. A gap in any
channel caps the composite regardless of the others.

### 3.2 THE DECISIVE FINDING — the named M1 acceptance statistic is UNSOUND
The audit blames one number for blocking 60% of weight:
`top1_token_unique = 1 across 16 distinct waves`. Two committed records contradict
its use as evidence:

* `m1_open_answer_gate.py` docstring: *"The prior verdict `top1_token_unique = 1`
  was VACUOUS: the random control scored 37/128 distinct vs treatment 39/128 —
  statistically indistinguishable."*
* `M1_EGRESS_RESOLUTION_20260916.md`: *"Distinctness of the top-1 token is
  ANTI-CORRELATED with content on this substrate. It does not measure semantic
  capacity; it measures dispersion. A gate built on this statistic cannot
  distinguish a working egress from noise, and in the measured case would have
  REWARDED noise."*

A statistic a near-orthogonal control can win is not a gate. Consequence: the
"blocks 60% of weight" claim is **not established by that measurement**, and the
correct status is `UNMEASURED`, not `BLOCKED_SEMANTIC_CAPACITY`.

### 3.3 The replacement metric EXISTS and is pre-registered
`m1_open_answer_gate.py` (10,523 B) is a five-arm design declared before
measurement: P1 determinism (100%), P2 distinct >= 0.50, P3 order-sensitivity
>= 0.50 (same multiset, shuffled), P4 equivalence >= 0.50 (whitespace-normalized
near view), and **P5 vacuity** — if the RANDOM-WAVE arm also clears P2 the gate is
declared vacuous no matter how good the treatment looks. That is the correct
falsifier for the defect above.

M1 status is therefore: the instrument is repaired, the gate has not been run at
N >= 100. `A2 RETRAIN_REQUIRED`, `M5 REQUIRES_APPROVAL`, `M6 BLOCKED`.

### 3.4 What is ACTUALLY missing (audited from tracked paths)
| Need | State |
|---|---|
| Score-eligibility schema | **PRESENT** — `henri_benchmark_registry.py` (12,259 B): `RunEvidence`, `BenchmarkRegistry`, `validate_score_eligibility`, `detect_synthetic_content`. EXTEND, do not fork |
| Isolated code-exec harness | **PRESENT but off-target** — `execute_authentic_coding_benchmark.py` runs HumanEval, which is NOT an AAII v4.3 member. `mbpp_secure_executor.py` + `s1_scicode_scaffold_runner.py` exist; SciCode is the right target (10% weight) |
| Live AAII runner | **QUARANTINED, reason known** — `_archive/invalid_evaluators/README.md`: the gauntlet "was moved here because its live imports do not match the current `zone_c_epistemic_axiom_harness.py` API". An API-drift repair, not a fraud finding |
| Tracked run evidence | **0 files** (`git ls-files '*run_evidence*'`). The audit cites 8 populated files; none are tracked, so no score claim is citable from the repo |
| Terminal-Bench 4.0 | **ABSENT** (`*terminal*bench*` -> 0 files). 10% weight |
| RUNNER-LEVEL LOADED-checkpoint gate | absent (architecture catalog) |

### 3.5 Ordered path to non-zero
1. **Repair the quarantined gauntlet's imports** against the live
   `zone_c_epistemic_axiom_harness.py` API, then re-gate it. Cheapest decisive
   step: it is an API-drift fix, not new capability.
2. **Run M1 at N >= 100** with the five-arm gate. If P5 holds (random arm below
   the floor) and P2/P3/P4 clear, the 40% open-answer channel becomes claimable.
   If P2 fails with P5 holding, M1 is `FALSIFIED_NO_EGRESS` — a clean negative.
3. **SciCode at scaffold scale** (<=16 subproblems) — the cheapest decisive
   coding test, CPU-feasible, 10% weight.
4. **Track run evidence**: every run writes `run_evidence.json` under a path keyed
   to the commit SHA, per the registry schema. Without this, no score is citable.
5. **Terminal-Bench 4.0** adapter (10%).
6. Everything else emits `BLOCKED_PRIVATE_GRADER`. Never silently omit a member.

## 4. Evidence classes
`OBSERVED` — all `git ls-remote`/`git show`/`ast`/pytest results and the file
inventory above. `DERIVED` — the "unsound statistic" interpretation (two committed
records agreeing). `BLOCKED` — the 75% externally graded set, and AAII itself
until an official harness produces a score. No AAII score is claimed.

## 5. Pre-registered kills
* `K-A FALSIFIED_NO_EGRESS` — P2 distinct <= floor with P5 non-vacuous.
* `K-B BLOCKED_INFRA` — any infrastructure error at scaffold cost.
* `K-C BLOCKED_PRIVATE_GRADER` — the 75% external set.
* `K-D BLOCKED_CONTAMINATION` — voids the run.
* `K-E PHANTOM_GATE` — any gate whose statistic a near-orthogonal control also
  clears is unsound and is retired rather than tuned (this is the M1 lesson).
