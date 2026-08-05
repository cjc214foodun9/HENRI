# Run 21 Pre-registration — Structured qFHRR Codec

## Evidence boundary

Run 20 measured a structureless `encode_text` path on the remote CUDA target at commit `40ab0b0`. The exact control reported distinct-string similarities near the random baseline `1/sqrt(65536) = 0.00390625`. This is `OBSERVED` Run 20 evidence. It does not prove that a replacement codec will work.

Run 21 tests one bounded hypothesis. It is not a full MBPP score run and it does not authorize a production-default codec change.

## Ground-state receipt

- Worktree: `C:\Users\chan\henri-worktrees\mbpp-heldout-v1`
- Branch: `mbpp-heldout-v1`
- Base HEAD before Run 21 edits: `455bb5a`
- Preserved unrelated untracked artifact: `henri_audit_chain.json`
- No staged changes before this manifest
- Run 20 CUDA baseline: A_EDMD ranks `{14: 6, 62: 42, 89: 42}`; B_SINGLE_PASS ranks `{14: 9, 62: 43, 89: 39}`

## Hypothesis

A fixed-width structured qFHRR encoder can preserve aligned character/token structure when it represents each character with a deterministic atomic phase engram, binds it to a deterministic position ring by modular phase addition, and bundles the bound phasors in the complex phase domain.

The hypothesis is falsifiable. A non-random geometry control is necessary but not sufficient for task ranking.

## Implementation boundary

- New module: `HENRI V2/qfhrr_structured_codec.py`.
- Tokenizer: character-level tokens. No merge vocabulary is fitted on MBPP items, test assertions, or canonical solutions.
- Atomic codebook: deterministic fixed-seed CPU-generated qFHRR rings for the supported character alphabet.
- Position codebook: deterministic fixed-seed CPU-generated rings, extended deterministically for longer inputs.
- Binding: `(q_position + q_token) mod 256`.
- Bundling: phase-domain vector addition followed by phase quantization to `uint8`.
- Output contract: one-dimensional `uint8` tensor of length `65536`, on the requested device, with values in `[0,255]`.
- Default path: unchanged SHA-256 full-string codec when `--codec legacy` is selected or the flag is omitted.
- Run 21 path: selected explicitly with `--codec structured`.
- Scope: rank probe only. Do not modify the MBPP pilot, production ARC path, Zone C schema, or archived code.
- Persistence: no task-specific Zone C writes and no database dependency.

## Exact controls

The probe must emit the Run 20-compatible fields:

- `identical_input_sim`: `f(3, 3, 3)` versus itself.
- `nearby_input_sim`: `f(3, 3, 3)` versus `f(4, 4, 4)`.
- `commutative_sim`: `return a + b` versus `return b + a`.
- `identical_output_sim`: `27` versus `27`.
- `nearby_output_sim`: `27` versus `28`.
- `random_baseline`: `1/sqrt(65536)`.

Additional diagnostics:

- same-token different-position similarity;
- unrelated-string similarity;
- codec name and version;
- tokenization and position-binding metadata.

## Pre-registered control gate

`control_healthy` requires:

1. identical controls are at least `0.999`;
2. `nearby_input_sim > 10 * random_baseline`;
3. `nearby_output_sim > 10 * random_baseline`;
4. `unrelated_sim <= 10 * random_baseline`;
5. `a+b` and `b+a` are not identical, proving that position binding is active.

If this gate fails, verdict is `INVALID_PLUMBING` and ranks are not interpreted.

The exact control values are implementation-dependent. The thresholds are the acceptance gate; they are not claims that the values must equal a theoretical closed-form result.

## Rank gate

Run the expressible subset `{14, 62, 89}` with both Run 20 variants:

- `A_EDMD`: per-item EDMD fit.
- `B_SINGLE_PASS`: per-item single-pass associative projection.

For each variant, record absolute rank and classification:

- `SELECTION_HIT`: rank `<=12`.
- `WINDOW_HIT`: rank `13..24`.
- `RANK_MISS`: rank `>24`.
- `PROBE_ERROR`: execution or implementation error; the run is not eligible for interpretation until repaired.

## Verdict criteria

- `ACCEPTANCE_MET`: `control_healthy == true` and, in at least one variant, both tasks 62 and 89 have rank `<=24`.
- `FALSIFIED_AT_SCALE`: `control_healthy == true` and either task 62 or 89 remains above rank 24 in both variants. Phase B grammar expansion remains blocked. The candidate-space attribution is a hypothesis, not a theorem.
- `INVALID_PLUMBING`: control gate fails, codec is not used at every probe call site, or legacy and structured codec instances are mixed.
- `BLOCKED_INFRASTRUCTURE`: CUDA execution, artifact retrieval, or required evaluator path fails. Do not convert this to a task miss.

Task 14 is reported but is not an acceptance driver. Its Phase A grammar and CEGIS escalation can produce low ranks without proving useful W_task compilation.

## Required artifacts

- `codec_geometry_control.json`
- `rank_probe_results.jsonl`
- `rank_probe_summary.json`
- complete stdout/stderr log with real exit code
- remote commit and clean-checkout receipt
- artifact SHA-256 digests
- synchronized bundle under `G:\\My Drive\\HENRI_Telemetry\\mbpp_run21\\`

## Remote execution

The benchmark must run on the Vast CUDA target, not local CPU. The local CPU run is a path smoke only. Remote execution must use a clean checkout of the pushed commit, one active GPU run, `setsid nohup`, a bracket-safe process check, and a `RUN21_DONE` marker.

## Reference context

The HENRI NotebookLM consultation is `INFERRED` design context only. It states that token codebooks require an external decoder/evaluation and that commutative positional binding has capacity limits. Live source inspection and CUDA telemetry override that consultation.

## Status

Pre-registered before Run 21 source implementation. No acceptance claim is made by this document.

## Addendum (2026-08-04, STRUCTURED_CODEC_KILL_V1_APPROVED)

- Arms added: `--codec legacy|structured|structured-nopos|structured-shuffled|identity`.
  `identity` = prompt-wave ranking with no W_task and no EDMD; its probe rows use
  variant `IDENTITY`.
- Geometry control extended: `position_swap_sim`, `unrelated_sim`, and codec metadata
  (name/version/tokenizer/position_mode).
- Matrix runner: `experiments/verification/run21_codec_matrix.py` (remote CUDA only).
- Contract tests: `tests/contract/test_qfhrr_structured_codec.py`.
- Verdict: `ACCEPTANCE_MET` = structured arm control_healthy AND (62,89) rank <= 24 in
  at least one variant AND improvement over both identity and legacy arms for the same
  variant; `FALSIFIED_AT_SCALE` = control_healthy but no acceptance; `INVALID_PLUMBING`
  = control gate fails; `BLOCKED_INFRASTRUCTURE` = CUDA/dataset/artifact failure.
- Identity arm uses `use_prompt_contrast=False` (plain wave similarity; no
  pred-minus-prompt subtraction that would be NaN when pred == prompt).
- Runner writes `RUN21_DONE` when the verdict is not infrastructure-blocked.

## Result (2026-08-04, remote CUDA, RTX 5090, commit 440f11d, worktree run21-wt2)

Verdict: `FALSIFIED_AT_SCALE`. All 5 arms exit 0, RUN21_DONE present,
control_healthy=true for the structured arm.

Per-arm ranks (task 14 / 62 / 89):
- legacy: A_EDMD 6/42/42, B 9/43/39 (exact run20 reproduction)
- structured: A_EDMD 10/30/7, B 6/52/13
- structured-nopos: A 10/31/5, B 1/20/40
- structured-shuffled: A 3/29/9, B 3/26/33
- identity: 9/46/54

Geometry controls (structured): nearby_input_sim 0.62 vs legacy -0.0045
(10x baseline gate passed); position_swap_sim 0.0066 (full) vs 1.0 (nopos) —
order sensitivity active; shuffled degrades order discrimination.

Acceptance required (62,89) rank <= 24 in one variant AND beats identity and
legacy. Structured best = A_EDMD 89=7 but 62=30 > 24; B 62=52. No variant
met the gate. FALSIFIED_AT_SCALE per prereg.

Attribution (HYPOTHESIS): structured geometry restores 89 (42->7) but 62
remains occluded (30-52) — consistent with run19's per-task phase occlusion
finding; a single global codec does not resolve task-specific occlusion.
Phase B grammar expansion remains blocked per prereg.

Evidence: G:/My Drive/HENRI_Telemetry/mbpp_run21/ (29 files, summary
sha256 f98e718bc491838c51eb844cbe3ada49ae6e0a0e93b8375811801eddabce63a2).
Commit: 440f11dcf1ed3b88b398a5e337fb182d672f3b6d.
