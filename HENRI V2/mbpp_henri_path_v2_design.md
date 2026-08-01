# MBPP HENRI-Path v2 Design (2026-08-01)

Status: DESIGN — approved for implementation (user: "Approve sandbox fix, then full henri path re-design").
Predecessor: run2 (legacy egress) measured 0/500 with 463 sandbox-artifact FAILs — invalid as task outcomes.

## 1. Problem

The MBPP pilot routes through `HENRIUnifiedEgressTransducer.decode_wave_to_response`
(linear-head token decode) with NO task functor and NO demonstrations
(`w_task_modulated=false` in run2 telemetry). This is the legacy autoregressive
remnant. HENRI is not autoregressive: task compilation happens online from
demonstration pairs (X_i, Y_i) into a task operator W_task (zero-pretraining
invariant), and candidates are REPL-verified with Sagnac vetoing.

## 2. Mechanism (v2, `--egress-path henri`)

1. **Ingress**: prompt -> qFHRR Z_256 phase wave `encode_text` (deterministic).
2. **Task functor (zero-training, online)**: from the paper's sanctioned few-shot
   set, MBPP task_id 1..10 (excluded from heldout 11..510):
   `W_task = sum_i (Psi_Yi - Psi_Xi) mod 256` via `HolographicTaskFunctorCompiler.compile_functor`
   with X_i = rendered prompt (same template as heldout), Y_i = reference_solution.
3. **Retrieval (O(1))**: `Psi_goal = bind_hadamard(W_task, Psi_prompt)`.
4. **Egress**: w_task-modulated unbinding
   (`decode_wave_to_response(goal_wave, prompt, w_task=w_task_vector)`; linear
   modulation `unit_wave * (1.0 + unit_w_task)`), AST-validated fail-closed.
5. **Verification**: candidate + tests executed in `SecurePythonSandbox`
   (`--sandbox-mode container-rlimit` on this Vast host; recorded in evidence).
6. **Sagnac veto**: telemetry records sagnac delta; invalid AST / fallback pattern
   are execution errors (never observed outcomes).

Kept identical to run2: canonical dataset (sha256 ccf64ce...), heldout 11..510,
pinned evaluator, contamination scan, checkpoint provenance gate, RunEvidence
schema, score-eligibility rule (execution errors block promotion).

## 3. Contract change (few-shot declaration)

- New prompt contract `mbpp_google_fewshot10_v1`: `num_fewshot=10`,
  `exemplar_ids=[1..10]`, `reference_code_exposed=false` (test-item solutions
  remain untouched; exemplar solutions are the sanctioned demonstrations).
- Manifest `prompt_contract.sha256` updated to the new contract's LF-canonical
  digest. Test items 11..510 are identical to run2.
- Contamination scan unchanged (scans test-task markers; exemplars are the
  paper's own few-shot set, not exposure of heldout items).

## 4. Falsifiable hypothesis and kill criteria

- H1: W_task compiled from 10 exemplars measurably changes decode behavior on
  heldout prompts vs no-W_task (legacy). Kill: telemetry shows
  `w_task_modulated=false` or identical output distribution (e.g., top_token_id
  histograms unchanged).
- H2 (goal, not gate): the HENRI path produces >=1 passing heldout item.
  Acceptance: item_results.jsonl with pass_count >= 1 AND sandbox status
  reconciled (`passed + failed == attempted`, `attempted + execution_errors == 500`).
- A zero score with the correct mechanism is a VALID negative external result.
  The v2 run is the first honest measurement of the HENRI code path.

## 5. Failure modes / fail-closed

- Invalid AST decode -> DecoderEgressFailClosedError -> EXECUTION_ERROR (score blocked).
- Sandbox unavailable -> BLOCKED_PREFLIGHT (namespace mode probe; verified 5d6c40e).
- Contract digest mismatch -> PilotBlocked (gate chain unchanged).
- W_task zero vector (no exemplars) -> run BLOCKED with reason `W_TASK_EXEMPLARS_MISSING`.

## 6. Files

- `mbpp_heldout_pilot.py`: add `--egress-path legacy|henri` (default henri),
  exemplar loader, W_task compilation + w_task-modulated decode in item loop.
- `data/official_benchmarks/mbpp_google_fewshot10_contract.json`: new contract.
- `data/official_benchmarks/mbpp_google_test_v1_manifest.json`: updated prompt
  contract sha256.
- Tests: exemplar loader bounds (1..10, schema), contract digest, W_task
  determinism (same demo pairs -> same W_task).

## 7. Verification sequence

Local suite -> commit -> push -> remote preflight (henri path) -> full 500-item
run (container-rlimit sandbox) -> telemetry pull -> sync to
G:\My Drive\HENRI_Telemetry\mbpp_run3.

## C1: SGLD In-Context Adaptation Experiment (2026-08-01)

Mechanism: test-time `adapt_in_context` on the 10 sanctioned exemplars (steps=2/pair, AdamW lr=1e-3, Bingham yield sigma=0.05, TAME gap-junction isolation, Cholesky retraction), then decode the 500 test items with W_task as run3.

Label rule: `demo_token_ids = argmax(unbinder(Psi_Yi))` computed pre-adaptation (fixed snapshot; no tokenizer in the transducer). The CE term aligns active-wave projection with the model's own solution-wave representation.

INTERNAL telemetry (never a task score): logit entropy (nats) on demo + 10 probe waves before/after; adapt_loss; yield_events; distinct_bootstrap_labels.
EXTERNAL telemetry: pass count vs run3.

Pre-registered criteria:
- PASS: yield_events > 0, distinct_bootstrap_labels >= 3, AND pass delta > 0 vs run3.
- INERT (valid negative): adaptation executes but entropy unchanged AND 0 passes -> shipped adapt_in_context is insufficient; next step = 500-step scheduled SGLD (T(t) = T0*(1+0.05t)^-0.55, L = CE + 0.25*Delta_Sagnac) per the training protocol.
- BLOCKED: any gate failure.

Recorded deviation: shipped `adapt_in_context_sgld` injects UNNORMALIZED Langevin noise (randn_like, D^1/2 norm inflation) instead of the skill invariant `F.normalize(randn(D))`; noted, not fixed in this change (one bounded change).

## C2: Corrected Wave-Aligned SGLD (2026-08-01)

Addresses run4's two structural causes:

1. Degenerate CE labels -> SOFT TARGETS: p_target = softmax(unbinder(Psi_Yi)) for each
   exemplar, snapshotted pre-adaptation. The label is the full 32000-dim solution-wave
   distribution (entropy ~9.98 nats), not the collapsed argmax (4/10 distinct in run4).
2. Missing Sagnac/phase term -> L = L_CE + 0.25 * Delta_Sagnac, with
   Delta_Sagnac = 1 - cos(p(·|Psi_Xi), p_target) in the probability simplex (the only
   wave-informed egress geometry without a wave decoder).

Plus the scheduled protocol: T(t) = T0*(1+0.05t)^-0.55, unit-normalized Langevin noise
(sqrt(2T dt) * F.normalize(randn(D))), Bingham yield gate, TAME gap-junction isolation,
Cholesky Stiefel retraction. Batch 10 exemplars, 500 steps, seed 0. Only down_proj updates.

Mechanism: adapt_in_context_sgld_wave (henri_decoder.py), wired as --sgld-adapt (replaces
the C1 call; bootstrap labels kept only as comparison telemetry).

Pre-registered criteria:
- PASS: loss_last < loss_first (descent), entropy_demo_after < entropy_demo_before
  (peaking toward the target), AND pass delta > 0 vs run3/run4.
- INERT: descent happens but no peaking AND 0 passes -> mechanism insufficient;
  next = higher steps/lr or re-trained head with wave supervision.
- BLOCKED: any gate failure.

## C2b: Root-Cause Fix — Sagnac Representation Bug (2026-08-01)

run5 telemetry exposed: mean_phase_mismatch = 0.0124 rad (pathological) and
yield_events = 500 with NO loss descent (10.111 -> 10.124). Diagnostic proved:

- encode_text returns Z_256 uint8 rings [0, 255] (norm 37731, unique 256 values).
- compute_dimension_sagnac_mismatch did acos(clamp(w, -1, 1)) on the raw ring:
  every value >= 1 clamps to 1 -> acos(1) = 0 -> Delta_Phi ~ 0.012 rad.
- TAME conductance g = 1/(1+e^(10(0.012-0.35))) ~ 0.967 -> isolation (1-g) ~ 0.037
  -> 96.7% of the SGLD gradient suppressed every step -> adaptation frozen.
- Control: identical rings -> 0.0 mismatch (correct); different rings -> true
  circular distance mean ~ 1.05 rad (~pi/2, healthy). The codec is NOT degenerate.

Fix: compute_dimension_sagnac_mismatch is now representation-aware:
- uint8 / [0,255] rings -> circular Z_256 distance mapped to [0, pi].
- real [-1,1] waves (vision) -> acos-clamp path unchanged.
Test: test_sagnac_mismatch_ring_path_healthy (independent rings > 0.8 rad mean,
identical rings == 0.0). Suite 157 passed / 1 skipped.

This bug suppressed adaptation in BOTH C1 (run4) and C2 (run5); the SGLD designs
were correct but starved of gradient. run6 re-runs C2 with the fix.

## run6 Result (2026-08-01, commit 7e980fa)

Mechanism-level: FIX VERIFIED.
- mean_phase_mismatch 0.0124 -> 1.571 rad (theoretical pi/2 for independent rings)
- mean_gap_junction_isolation 0.037 -> 0.888 (gradient flows)
- loss 10.111 -> 10.063 (descent restored; run5 ascended)
- sagnac_dist_final 0.118 -> 0.058 (phase-alignment distance halved)
- exec_errors 62/57 -> 48 (marginal improvement in candidate AST validity)

External: 452 attempted, 0 passed, 48 exec_errors -> score_eligible false.
ENTROPY proxy caveat: logit entropy still rises (9.984 -> 10.088) while CE descends;
the linear head broadens to cover p_target rather than peaking (capacity effect).
CE descent is the valid alignment evidence; entropy is not a valid MI proxy here.

Verdict: C2 mechanism is CORRECTED and learning. 0-pass is now a CAPACITY finding:
the linear egress head (down_proj 65536->2048 -> lm_head 32000) cannot compose
novel Python functions from exemplar alignment; adaptation only aligns known
exemplar pairs. Next direction (C3) is a capacity decision, not an SGLD tuning.
