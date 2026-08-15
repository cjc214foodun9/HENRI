# Phase 8.7 — Typed Action Embeddings & Valence-Free Pre-Training (PRE-REGISTRATION + AUDIT)

Source PDF (Drive inbox, two identical copies): `HENRI V2 Phase 8.6 Final Postmortem & Phase 8.7 St....pdf`
raw SHA-256 `97d46428d88988e940f548f4688b3a02b751d8e17a7ae9db86d693098b8997a7`
(extracted text LF SHA `7c322bca…`, 5 pp, 10,741 chars — extracted 2026-08-14, protocol executed 2026-08-14).
The PDF confirms the sealed Phase 8.6 state exactly (Run 3 verdicts @ `667a74f`, seal `25e673e`, main `2218ec4`).

## Protocol audit (against live main @ `2218ec4`, fresh worktree)

| Step | Protocol command | Audit result |
|---|---|---|
| 1 | `git checkout -b feat/typed-action-phase-embeddings` from main `2218ec4` | DONE — worktree `C:/tmp/henri-87-wt`, branch `feat/typed-action-phase-embeddings` @ `2218ec4`, zero status lines |
| 2 | `python HENRI V2/wave_jepa.py --mode typed_action_test` | **BLOCKED — phantom CLI.** `wave_jepa.py` has 0 argparse/`--mode` hits; its `__main__` is a random-grid toy demo. Dead code: zero production callers (only `_archive/invalid_evaluators/henri_benchmark_gauntlet.py` and `_archive/orphans/` reference `WaveJEPA`). Same misdirection family as the D1/D2 and Phase 8.6 PDFs. Production transition = `LowRankCoupledTransition` in `efe_planner.py`. |
| 3 | `python HENRI V2/physical_world_model_benchmarks.py --mode valence_zero` | **BLOCKED — wrong path + no such mode.** File lives at `HENRI V2/experiments/exploratory/physical_world_model_benchmarks.py`; its argparse is `--scale production\|reduced --device cuda\|cpu`. No `--mode`, no `valence_zero`. |
| 4 | `pytest HENRI V2/tests/ -v` | DONE — **405 passed, 1 skipped, 14 warnings in 30.37s** (env-clean local Py3.14, `PYTHONPATH="HENRI V2"`, worktree @ `2218ec4`) |

Verdict: steps 2–3 cannot run as written; the levers they verify are NOT implemented. Executing the
protocol verbatim would either fail (step 3: file not found / unrecognized arg) or run a meaningless
toy demo (step 2). NO fabrication; NO fake verification runs.

## Pre-registered levers (from the PDF; each default-OFF, one bounded change)

- **Lever 8.7-A — Typed Continuous Action Embeddings (P0):** actions as Clifford multivector phase
  rotators `Ψ_a ∈ S^{D-1}`, bound via non-commutative Clifford product `Ψ_bound = Ψ_t ⊗_Clifford Ψ_a`.
  PDF gate: held-out Sagnac loss `< 0.15` over 32 evaluation trajectories.
- **Lever 8.7-B — Valence-Free (ν=0) Pre-Training (P0):** decouple transition training from
  exteroceptive reset penalties; train pure forward wave dynamics on continuous physical trajectories.
  PDF gate: transition error `< 0.10` across 50 un-docked physical environment steps.
- **Lever 8.7-C — Transition Persistence & Coordinate Equivariance (P1):** maintain Lie-group spatial
  transformations continuously across trajectory steps. PDF gate: cross-step phase coherence `r > 0.90`
  over rollouts `H ≥ 10`.

## Promotion criteria (PDF; unchanged)

1. Held-out Sagnac loss reduction > 15% under typed continuous action embeddings.
2. Transition prediction error `L_Sagnac < 0.10` on un-docked physical control environments.
3. In-situ update cycle ≤ 45.0 ms at D=65,536 on RTX 5090.

## Discipline

- Implementation of 8.7-A/B/C happens ONLY in the PRODUCTION path (`efe_planner.py` transition +
  `physical_control_environments.py`), default-OFF named flags, one bounded change at a time.
- No edits to dead `wave_jepa.py`; do not add a CLI to dead code (misdirection trap).
- Pre-registered kill: any gate fires ⇒ lever sealed FAIL; no post-hoc threshold tuning.
- Remote CUDA verification on the Vast 5090; local CPU runs are NOT verification.
- Branch `feat/typed-action-phase-embeddings` from main `2218ec4`; main untouched; NO promotion.
- Status: **AWAITING explicit scope approval to implement levers 8.7-A/B/C** (verification commands
  in the PDF are post-implementation gates, not runnable against current main).
