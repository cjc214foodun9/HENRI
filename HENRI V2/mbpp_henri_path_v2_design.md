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
