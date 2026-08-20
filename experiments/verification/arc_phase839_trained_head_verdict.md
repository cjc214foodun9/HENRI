# Phase 8.39-V3 — Execution-grounded correctness head verdict

Status: **FALSIFIED (pre-registered Gate A fired: /35 correct rank 15 > 12)**
Branch: `phase839/humaneval-wave-ast` @ `2abf70f`
Date: 2026-08-20

## Experiment (approved option 3, sequential after Phase 2 BLOCKED)

Trained a linear correctness probe `w ∈ S^{D-1}` over the decoder's relative-direction feature
`v_rel = normalize(wave(body) − cos·wave(prompt))` on **MBPP** (google-research canonical,
563,743 B, sha256 `ccf64ceae9c5403b`, 974 items) — a DIFFERENT benchmark, so HumanEval stays
unseen (zero-leakage). Labels from **real sandbox execution** (container-rlimit mode; namespace
blocked by seccomp — fail-closed): positives = gold code passing `test_list`; negatives =
grammar candidates + perturbations failing all tests.

## Training (OBSERVED, CUDA RTX 5090, D=65,536)

- Labels: pos=60, neg=480, skip=0 (bounded kill-gate scope `--max-items 60`)
- 40 epochs AdamW lr 1e-3: loss 0.6931→0.5237, trn_acc 1.000, **val_acc 0.6019**
- Checkpoint `HENRI V2/models/correctness_head_839.pt` sha256 `e80d0579f6541bd2`
- Feature = same v_rel used by `decode()`; head applied as `--trained-rank` (default-OFF)

## Gate A oracle (OBSERVED, CUDA, full 71-candidate sets)

| Item | correct body | enum rank (baseline) | trained-head rank | in window ≤12 |
|---|---|---|---|---|
| HumanEval/23 | `return len(string)` | 3 | **6** | ✓ |
| HumanEval/35 | `return max(l)` | 5 | **15** | ✗ |

Top-5 under trained head (both items) are plausible-but-wrong bodies (`return int(string, 16)`,
`return sum(l) // len(l)`, …) — the probe captures weak lexical-shape preference, not correctness.

## Pre-registered gates

- Gate A: both /23 and /35 correct bodies rank ≤ 12 → **FAILED** (/35 = 15). Kill fires.
- Gate B (full run ≥ 3/50): not run — the predictable outcome under Gate A is ≤ 1/50
  (both baseline passers displaced), so running it would burn GPU on a killed lever.
  Predicted value is `DERIVED`, not `OBSERVED`.

## Mechanism (DERIVED)

A single linear direction over the random-ring qFHRR program-wave superposition cannot
separate correct from incorrect full-program waves: 540 execution labels vs the
non-compositional codec's near-orthogonal geometry (run20 finding) leaves no semantic axis
to learn. val_acc 0.602 ≈ chance + 0.10 corroborates.

## Disposition

- `--trained-rank` stays default-OFF, unpromoted. Checkpoint preserved as evidence.
- Ranking-lever class now CLOSED with 5 falsified mechanisms: reward-rank (1/50),
  decoder-rank (0/50, oracle 49/71+68/71), spec-rank (1/50), trained linear probe
  (Gate A fail), codec position modes (all < 0.31 gate).
- Standing scores unchanged (all OBSERVED): HumanEval 2/50, GPQA 0.298, MMLU 0.2598.
- Next lever requires a representation change (structured-codec program waves / nonlinear
  head / supervised code-wave codebook with MBPP pretraining) — new experiment, new
  pre-registration, requires approval. Re-proposing the killed linear probe without new
  evidence stays FALSIFIED.

## Evidence artifacts

- Oracle: live CUDA python (ranks above; log `/root/telemetry_logs/train_correctness_head_839.log`)
- Checkpoint: `HENRI V2/models/correctness_head_839.pt` (sha `e80d0579…`)
- MBPP: `HENRI V2/data/mbpp.jsonl` (sha `ccf64cea…`)
- Governance event appended.
