# Phase 8.39 — Class 3.0 Discriminative Phase Representation verdict

Status: **FALSIFIED at pre-registered proxy Gate A (treatment = both levers)**
Branch: `phase839/humaneval-wave-ast` @ `6305e45` (base)
Spec: `HENRI_Class_3.0_Discriminative_Phase_Representation_Plan.md`
Date: 2026-08-20

## Premise and construction (OBSERVED)

- Class 3.0 levers were NOT present in live code (0 grep hits for
  `--codec-carrier-subtract` / `--ast-idf-weighting`). Implemented faithfully
  from the spec:
  - `HENRI V2/qfhrr_ast_discriminative_kernel.py` — `ASTDiscriminativeEncoder`:
    Lever 3.1 Gram-Schmidt carrier projection in complex phasor space then
    `Quantize_256(arg(...))`; Lever 3.2 IDF weighting
    `w(op) = log(1 + N/f(op))` at phasor accumulation. Both default-OFF.
  - Carrier + node histogram compiled from canonical MBPP (974 records, sha
    `ccf64ceae9c5403b`; 76 node types; top: Name 12192, Load 11260, Constant
    3653, Store 3131, BinOp 2086, Call 2013).
- Probe `experiments/verification/arc_phase839_class3_gate_a_probe.py` reuses
  the production 71-candidate grammar pool (`WaveASTDecoder._instantiate`,
  same tables) for HumanEval/23 (strlen, 1 arg) and /35 (max_element, 1 arg),
  ranked by mean raw phase-cosine vs the MBPP codebook (N=100) in the SAME
  configuration (Class 2.0 precedent).
- Corpus consult (NotebookLM, authenticated): Gram-Schmidt carrier subtraction
  and IDF weighting are ABSENT from the HENRI corpus — novel construction,
  spec doc is the only authority. Conflict note: the corpus frames carrier
  dominance via CC-OS background masking (a different mechanism).

## Proxy Gate A (OBSERVED, CPU d=2048; deterministic True on all arms)

| arm | E[cos] (M1 ≤ 0.10) | /23 rank (≤ 5) | /35 rank (≤ 5) |
|---|---|---|---|
| none (Class 2.0 control) | 0.4301 | 31 | 33 |
| idf only (Lever 3.2) | 0.3484 | **3** | **5** |
| carrier only (Lever 3.1) | 0.3451 | 36 | 38 |
| **both (gated treatment)** | **0.2526** | **10** | **12** |

- Treatment arm: M1 FAIL (0.2526 > 0.10), M2 FAIL (10, 12 > 5).
  **Kill fires. Gate B skipped. CUDA D=65,536 not run.**
- Direction argument (DERIVED): carrier overlap is set by deterministic shared
  node-type vectors, approximately D-invariant (Class 2.0 precedent); no arm
  approached M1 at the proxy, so no credible path to M1 at D=65,536.

## Mechanistic findings (INFERRED from OBSERVED arm contrasts)

1. **Carrier-only made ranks WORSE (31→36, 33→38).** For single-line return
   bodies, the "shared skeleton" nodes (Name/Load/Return/Call) ARE the
   semantic content; subtracting the global carrier removes the answer signal.
2. **IDF-only reached M2 (3, 5) while leaving cos at 0.3484** — un-pre-registered
   arm, recorded NOT promoted. It partially contradicts the spec's causal
   claim that carrier removal (cos drop) is NECESSARY for ranking. Evidence is
   weak (2 items, N=100 codebook, d=2048 proxy) — HYPOTHESIS for a future
   pre-registered Gate A' (IDF-only, M2-only), not a pass.

## Disposition

- Components committed as sealed experiment artifacts, default-OFF. No runner
  wiring (`--codec-carrier-subtract` / `--ast-idf-weighting` NOT added to the
  production runner): a killed path does not become a dead flag.
- Standing scores unchanged (HumanEval 2/50, GPQA 0.298, MMLU 0.2598 — OBSERVED).
- Governance event appended; registry updated.

## Evidence

- Probe: `experiments/verification/arc_phase839_class3_gate_a_probe.py`
  (run 2026-08-20, CPU, d=2048, exit 1 = FALSIFIED by design).
- Kernel: `HENRI V2/qfhrr_ast_discriminative_kernel.py`.
- Data: `mbpp.jsonl` sha `ccf64ceae9c5403b`; HumanEval cache (164 items).
