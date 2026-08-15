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

## APPROVED IMPLEMENTATION PRE-REGISTRATION (user go, 2026-08-14; option 1: 8.7-A + 8.7-B)

New module `HENRI V2/henri_typed_actions.py` (diagnostic-only, default-OFF; `efe_planner.py` NOT modified):

- **8.7-A `TypedActionEmbedding`**: action token k -> structured S^{D-1} wave
  `cos(omega_k * n) + sin(omega_k * n)` over flattened D=65,536, with
  incommensurate carrier `omega_k = 2*pi*(sqrt(p_k) mod 1)` (distinct primes).
  Reshaped [num_blocks, 8], per-block unit norm (planner boundary contract).
- **8.7-A `clifford_bind`**: non-commutative even-subalgebra Clifford
  (Hamilton quaternion) geometric product per block; order matters
  (bind(a,b) != bind(b,a)). Replaces commutative FHRR bind ONLY in the
  experiment arms.
- **8.7-B valence-free corpus**: un-docked InvertedPendulum/CartPole rollouts
  (no reset penalties, no reward coupling — nu=0), (Psi_t, a_t, Psi_{t+1})
  triples; production `train_transition_batch` fit on the accumulated corpus.

Gates (PDF, unchanged):
- 8.7-A: held-out Sagnac prediction loss < 0.15 over 32 eval trajectories;
  promotion additionally requires >15% held-out reduction vs random-action baseline.
- 8.7-B: transition error L_Sagnac < 0.10 across 50 un-docked physical env steps.
- Latency: in-situ update cycle <= 45.0 ms at D=65,536 on RTX 5090 (measured).

Remote matrix arms (paired discipline from Phase 8.6 — SAME state trajectories
across arms, disjoint train/eval seeds; per-seed metrics; DONE marker only if
all arms rc=0):
- A0 baseline: random action waves + production FHRR bind (current default path).
- A1: TypedActionEmbedding + FHRR bind.
- A2: TypedActionEmbedding + Clifford bind.
- A3 (8.7-B): valence-free pre-training; 50 un-docked steps; L_Sagnac < 0.10 gate.
- Latency probe: update cycle ms at D=65,536.

Known risk (pre-registered): `state_to_wave()` in physical_control_environments.py
is a SEEDED RANDOM projection (state hash -> randn -> normalize), i.e. the R1
representation bottleneck. The 8.7 diagnostic may therefore fail its gates
even with typed actions; that outcome seals FAIL honestly (no post-hoc tuning).

## SEALED VERDICT (2026-08-14) — Phase 8.7 (feat/typed-action-phase-embeddings)

Definitive run @ `03609d902971e1c494081b8407aa8843b9891a67`; remote CUDA
matrix @ RTX 5090 (torch 2.12.0+cu130, CUDA 13.0, D=65,536, num_blocks
8192); all arms rc=0; DONE_MARKER rc=0. Bounded CUDA smoke (HENRI_SMOKE=1)
at the same SHA ran all 5 arms rc=0 before the full matrix.

Evidence: `phase8_evidence/phase87_typed_actions/`
- `p87_result.json` SHA-256 `d956fc446dc8ab5f3fac2a0895c42918063dd3c9856266b14dac54a8632519c5`
- `p87.log` SHA-256 `ab7199d23a36389fa51761be37660c65e6c2c15a459ca6a011368ddd052edfe9`
- decoder checkpoint loaded: SHA `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060` (symlink overlay)

| Arm | Result | Key metrics |
|---|---|---|
| A0 baseline | OK | random action waves + production FHRR bind; held-out Sagnac 0.9987 (quasi-orthogonal floor ~1.0) |
| A1 8.7-A typed + FHRR | P87_FAIL | held-out 0.9991; reduction vs A0 **−0.04%** (no improvement) |
| A2 8.7-A typed + Clifford | P87_FAIL | held-out 0.9997; reduction vs A0 **−0.10%** (no improvement) |
| A3 8.7-B valence-free | P87_B_FAIL | held-out 0.9999 vs gate < 0.10 over 50 un-docked steps |
| LAT | LAT_FAIL | update cycle **294.2 ms** vs <= 45.0 ms gate (fwd 433.2 ms) |

Verdicts:
1. **Lever 8.7-A (typed action embeddings) FALSIFIED** — structured action
   carriers + non-commutative Clifford binding give ZERO held-out improvement
   over random action waves (delta -0.04% / -0.10%, slightly worse). Actions
   were not the bottleneck.
2. **Lever 8.7-B (valence-free pre-training) FALSIFIED** — transition error
   0.9999 >> 0.10 gate. Valence decoupling alone cannot train the transition
   on seeded-random state projections.
3. **Latency gate FAIL** — in-situ update 294.2 ms (6.5x over the 45.0 ms
   gate) at D=65,536; promotion impossible regardless.
4. **Attribution (INFERRED, corpus-confirmed)**: the state encoder
   `state_to_wave()` is a SEEDED RANDOM projection (state hash -> randn ->
   normalize) with NO spatial carrier structure — the R1 representation
   bottleneck. Corpus: with un-factored random state projections, typed
   actions and valence-free pre-training cannot help; held-out Sagnac clamps
   at the quasi-orthogonal floor ~0.999-1.0. Prescribes representation
   repair (CC-OS foreground-masked spatial carriers, incommensurate P_x/P_y
   bases) as the NON-NEGOTIABLE next lever before any action featurization.
5. This outcome was PRE-REGISTERED as a known risk above (lines 81-84):
   "state_to_wave() is a SEEDED RANDOM projection... the 8.7 diagnostic may
   therefore fail its gates even with typed actions; that outcome seals FAIL
   honestly." The sealed failure CONFIRMS the pre-registration.

Branch sealed @ `03609d9`; `main` untouched `2218ec4`. NO promotion.
Next lever: representation repair (R1 foreground-masked ramps) — the
corpus-prescribed non-negotiable prerequisite; requires a new pre-registered
protocol and explicit approval.


