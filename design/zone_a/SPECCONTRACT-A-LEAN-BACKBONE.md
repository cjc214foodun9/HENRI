# SpecContract A — lean backbone for a 32 GB RTX 5090 (APPROVED)

Status: **APPROVED** by the operator on 2026-10-03. Approval event
`#10b2d1f45cd8626c` in the hash-linked ledger (chain 1816 → 1817).
Scope of approval: the A-K4 egress kill plus MVP wiring, local and $0.
Latency claims are explicitly **DEPRIORITIZED** by the operator. The goal is a
working HENRI MVP model, not a latency result.
This approval does **NOT** authorise training runs, backbone training, or GPU spend.

## A-K4 RESULT (2026-10-03) — ran and PASSED

Measured on a 6-fact synthetic corpus, 12 items, CPU, frozen
Qwen2.5-1.5B-Instruct (1,543,714,304 params, `unexpected_key_count=0`):

| arm | context delivered | hit rate |
|---|---|---:|
| A0 | none | **0.000** |
| A1 | correct retrieved chunk | **0.833** |
| A2 | mismatched chunk (excludes target source) | **0.083** |

`K4a +0.833 PASS · K4b +0.750 PASS · K4c PASS · K4d determinism 1.000 PASS`
Verdict `AK4_PASS_CONTEXT_IMPROVES_GENERATED_ANSWER`.
Receipt `design/zone_a/evidence/ak4_egress_receipt.json`,
sha256(LF) `8d731a1052015367c5c9aec496be28004fe81e85a9ba84e784d712b91df59ecb`.

### What this PASSES, and what it does NOT

**It proves ARM A**: retrieved context delivered to the frozen backbone as
**raw text** measurably and deterministically improves a generated answer.
That is retrieval VALUE established end to end.

**It does NOT prove ARM B**: Wave-to-text egress via Zone C wave
**conditioning**. The `top1_token_unique = 1` defect was measured on the ARM_U
unbinder path (`down_proj [2048,65536] → lm_head [32000,2048]`), and A-K4 as run
does not exercise that path. **ARM B remains UNMEASURED.**

This distinction is load-bearing. If Arm A works and Arm B fails, the defect is
the **bridge**, not Zone C retrieval — a different fix on a different timescale.
Do not read this PASS as clearing the wave-to-text bridge.

## CLOSED-VOCAB EGRESS PROBE (2026-10-03) — ran and FAILED

A third egress path exists that the `top1_token_unique=1` diagnostic did **not**
measure: `HoloEgressCodebook` (`henri_vla_tokenizer.py:433`), a CLOSED 156-word
vocabulary whose codebook is **derived from the tokenizer**, not random.
Probe: `experiments/verification/armB_closed_vocab_egress_probe.py`.
3 seeds x 3 carrier templates x 200-permutation null.

| Check | Rule | Result |
|---|---|---|
| Q1 content above null | rate > null_max, every template | **PASS** (0.32-0.63 vs null_max ~0.03) |
| Q2 specificity | mean margin own-index vs best-wrong > 0 | **FAIL** (+0.15, -0.16, -0.21, -0.18, -0.40, -0.25, -0.35, -0.08) |
| Q3 random control | control < treatment | **PASS** (control 0.000-0.006) |
| Q4 round-trip | bare-word identity == 1.0 | **PASS** (156/156 = 1.000) |

Verdict `CLOSED_VOCAB_EGRESS_FAIL:Q2_specificity_positive_margin`.
Receipt `design/zone_a/evidence/armB_closed_vocab_receipt.json`,
sha256 `a49620be91ddb9c3e5aad8a4d951369fd3030696ea1a2c4ad840782d7b32bfe3`.

**What this means.** Bare manifest words round-trip **perfectly** (156/156). The
same words embedded in a carrier phrase recover at only **0.32-0.63**, far above
chance but with a **negative mean specificity margin on most templates** — the
own-index logit does not reliably exceed the best wrong index. The readout is
**bimodal**: a subset of words is cleanly recovered, the rest are not recovered
at all. That is **not a working egress**.

**Position dependence observed (not yet a kill).** The only template with a
positive margin on two of three seeds places the target word **first**
(`{w} is the word`, 0.51-0.63). Mid-phrase templates (`the {w} report`,
`describe {w} now`) are negative on all seeds. Consistent with a
`position_binding="fractional_shift"` artifact, but this is an observation at
n=3 seeds and is **not** established.

**Scope.** This is NOT the open-vocabulary ARM_U unbinder path
(`down_proj [2048,65536]`, 32000 tokens) where `top1_token_unique=1` was
recorded. **ARM B remains UNMEASURED for open-vocabulary text.** A closed
156-word vocabulary is a far easier problem than 32000-token generation, so a
pass here would not have implied a pass there.
Evidence classes: OBSERVED (repo/measured), INFERRED (external sources, untested here),
DERIVED (arithmetic), HYPOTHESIS, BLOCKED.

## 1. The question this contract answers

The goal-state assessment (`GOAL-STATE-ASSESSMENT.html`, commit `2737800`) concluded:
HENRI lacks a vision encoder, a language backbone in the load path, an action decoder,
and a model-scale training loop. AAII v4.3 on a single 5090 is **not realistic as built**.

This contract specifies the smallest backbone that can close that gap **if** it is built,
and states honestly what it still would not achieve.

## 2. Constraint envelope (DERIVED)

| Quantity | Value | Source |
|---|---:|---|
| VRAM | 32 GB GDDR7 | pin |
| Memory bandwidth | 1792 GB/s | pin |
| NVFP4 tensor cores | native | pin |
| Host RAM | 31.1 GiB (measured) | OBSERVED |
| CUDA available locally | **False** (`torch 2.13.0+cpu`) | OBSERVED |
| Local quant libs | `bitsandbytes`, `peft`, `accelerate` **ABSENT** | OBSERVED |

**Consequence.** Weights must fit in 32 GB. KV cache must fit in the remainder.
This rules out dense models above roughly 31B at Q6.

## 3. Findings from primary sources (INFERRED — external, not reproduced here)

| Finding | Source class | Class |
|---|---|---|
| MoE with low active params dominates dense at equal VRAM on this card | vendor/community benchmarks | INFERRED |
| A 35B-A3B MoE fits ~22–28 GB at Q4/Q6 and is the commonly reported best fit | community benchmark tables | INFERRED |
| Hybrid Mamba-2 + attention + MoE ratios near 1:7 are the reported sweet spot for long context at reduced KV | paper summaries / model cards | INFERRED |
| A 26B MoE activating ~3.8B reaches ~97% of its 31B dense sibling at ~12% compute | vendor architecture write-up | INFERRED |
| Hybrid SSM+attention MoE is the current long-context efficiency pattern | survey synthesis | INFERRED |

**These are INFERRED, not OBSERVED by HENRI.** No benchmark in these sources transfers to
this repo. Community numbers are not evidence of HENRI capability.

## 4. Proposed architecture (HYPOTHESIS)

```
HENRI lean backbone — target 32 GB @ Q4/Q5
  token mixer   : hybrid. ~1 attention layer per 7 SSM layers.
                  attention = GQA (few KV heads) for a small KV cache.
  feed-forward  : MoE, top-k sparse. Active params ~3-6B of a 26-35B total.
  vision        : ViT or SigLIP-class encoder, trained in (not bolted on).
  action head   : continuous + tokenized policy, distinct from ARC grid heads.
  memory        : Zone C retrieval conditions a step (ALREADY BUILT).
  trainable     : none at inference. LoRA default-OFF for adaptation.
```

Four reasons this shape, each tied to a measured constraint:

1. **Sparse wins on bandwidth-bound hardware (INFERRED + DERIVED).** Decode is
   bandwidth-bound. Active params, not total params, set the cost. A MoE keeps
   quality while shrinking per-token compute.
2. **Hybrid mixing bounds KV cache (DERIVED).** Only attention layers scale with
   sequence length. Most layers are recurrence, so long context survives in 32 GB.
3. **Zone C is the memory layer, not the backbone (OBSERVED).** 65536-dim engrams,
   2000-dim HNSW, no corpus VRAM residency, bit-exact round-trip. It conditions a
   step; it does not generate tokens. Do not re-cast it as a language model.
4. **Zero-pretraining invariant (policy).** A frozen revision-pinned backbone is
   permitted (`HENRI_BACKBONE=1`, CLASS51) with matched ablations and a contamination
   review. Any contribution MUST be separated from the frozen backbone.

## 5. Pre-registered kills

| ID | Test | Reading |
|---|---|---|
| A-K1 | Backbone loads with zero newly-initialized params | gate already implemented in `henri_backbone_adapter.py`; missing/mismatched keys raise |
| A-K2 | Bit-exact output parity at fixed seed across two loads | mismatch ⇒ non-determinism, stop |
| A-K3 | Fits 32 GB at declared quant **with** declared context | OOM ⇒ shrink or drop a rung |
| A-K4 | Retrieved context must improve a **generated** answer: content-presence rate with correct evidence exceeds the no-evidence arm by ≥0.50 | the recorded egress defect: 16/16 chunks gave the same top-1 token |
| A-K5 | Matched ablation: Zone C conditioning ON vs OFF | no gain ⇒ memory layer does not transfer at this scale |

**A-K5 / ARM B RESULT (2026-10-03) — ran and FAILED. Negative result, retained.**
Implementation: `HENRI V2/experiments/verification/ak5_armb_egress_gate.py`.
Receipt: `design/zone_a/evidence/ak5_armb_egress_receipt.json`
(`receipt_sha256 a043ebacea9b3795aeeab717f4e70f51259b806c8a2a7849d7ee05f819095a20`).
Scope: read-only, no optimizer step, local CPU, $0. Artifact:
`henri_decoder_checkpoint.pt`, 799034119 B, sha256 prefix `75572389083455a3`
(identical in `aaii-v43` and `zone-a-selfplay`).

This gate measures the path the operator named — wave → text through
`HENRINeuralEgressUnbinder` (`d_model=65536, d_hidden=2048, vocab_size=32000`) —
not the text-splicing path A-K4 measured.

| Check | Measured | Pass |
|---|---|---|
| D0 artifact trained (`rel_delta` > 0.5) | 1.007 (down_proj), 1.146 (lm_head) | **yes** |
| Q1 separation vs same-family shuffled control | 0.519 vs 0.550 → **−0.031** | **NO** |
| Q1 separation vs random control | 0.519 vs 0.575 → −0.056 | NO |
| Q2 margin vs same-family shuffled control | 0.218 vs 0.171 → +0.047 (< 0.10) | **NO** |
| Q2 margin vs random control | 0.218 vs 0.280 → −0.063 | NO |
| Q3 determinism | bit-exact true | yes |
| Q4 rotation sensitivity | +0.201 / +0.265 / +0.284 | yes |

**Verdict: `AK5_EGRESS_FAIL:Q1_separation+Q2_margin`.**

Three findings, stated exactly:

1. **The artifact IS trained** (D0 passes). So the failure is not an untrained
   head, and the verdict is admissible.
2. **The wave DOES reach the head** (Q4 passes; rotation moves the margin by
   0.20–0.28). The bridge is wired and live.
3. **The readout does not carry content.** Content-destroyed *same-family* waves
   — 8192 blocks shuffled, preserving every block vector and unit norm — separate
   **as much or more** than real content-bearing waves (Δ = −0.031). When the
   control beats or matches the treatment, the measurement is not evidence of
   content. This is the same failure signature as the closed-vocab path
   (specificity margin −0.21): **the readout is dominated by bulk block
   statistics that survive content destruction.**

**Did not reproduce the recorded count.** The recorded defect is
`top1_token_unique = 1` across 16 distinct waves. This run measured
`top1_token_unique = 7` across 10 distinct waves (tokens 12043, 9237, 10882,
28697, 28163, 24381, 20832; two tokens repeat). Collapse is present (7 < 10) but
not total. The original count is **not reproduced here** and must not be quoted
as this run's result.

**Design consequence.** The failure mode is readout, not wiring and not training
absence. Per spec the next probe is **carrier suppression / readout
normalization** against the same shuffled control — not retraining, and not a new
egress architecture. Two independent egress paths now fail the same specificity
test, so a third untested path is not the next move.

**A-K4 METRIC AMENDED (2026-10-03) — evidence-based, disclosed.**
The original A-K4 wording above ("emits distinct top-1 tokens above floor") was
**falsified as a metric** by the pre-registered `m1_open_answer_gate.py`, first
executed 2026-10-03. Verdict for both position-binding arms:
`VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE`. The random-wave arm scored
`distinct_ratio` 0.59 (fractional_shift) and 0.70 (phasor_bind) while the
treatment scored 0.23 and 0.12 — the **negative control beat the treatment**.
So distinct-top-1 count is not evidence of semantic content; it is the known
`argmax is beta-invariant` defect. A-K4 therefore scores **content presence in
the generated answer**, with A0 (no evidence) and A2 (mismatched evidence) as
negative controls. The kill is unchanged in kind: context must improve the
answer. Only the metric changed, and it changed because a measured control beat
the treatment. Implementation: `HENRI V2/experiments/verification/ak4_egress_gate.py`.

**A-K4 is the binding one.** Wave-to-text egress currently does not discriminate
(`top1_token_unique = 1` across 16 distinct chunk waves). That blocks ~60% of AAII
weight on its own. No architecture fixes it; it is an egress problem.

## 6. What this still would NOT achieve (the honest ceiling)

Per the pinned v4.3 audit (`henri-research/references/aaii-v43-composite-and-exposure-audit.md`):

| Access | Weight | Members |
|---|---:|---|
| Locally reproducible | **25%** | SciCode, Terminal-Bench 4.0, AutomationBench-AA |
| Externally graded / private | **75%** | Elo panels, HLE, LCR, AA-Omniscience, CritPt, GDP.pdf |

**A local sub-index can cover at most 25% of the composite.** The AAII scoring harness
(`aaii_scoring_harness.py`) already enforces this: it reports
`local_weight_covered=0.25` and `BLOCKED_PRIVATE_GRADER` for the rest, and it
**refuses to emit an "AAII v4.3 score"**.

So even a perfectly built lean backbone yields a **DERIVED 25% sub-index**, never a
v4.3 composite. Claiming SOTA on v4.3 requires the operator's published result.

## 7. Blockers (BLOCKED)

| Item | Status | Evidence |
|---|---|---|
| In-tree autoresearch CLI | **BLOCKED** | `import agentic_graph` succeeds; `agentic_graph.autoresearch_cli` → `ModuleNotFoundError` in this checkout |
| Local CUDA for training | **BLOCKED** | `torch.cuda.is_available() == False`; `peft/trl/datasets` absent |
| Local 0.85 threshold on Jev advice | advisory only | `authorization: false`, `ESCALATE` |
| Honcho | offline by policy | do not activate |

## 8. Required next gate

1. Human acceptance or rejection of this contract.
2. If accepted: stage a pinned backbone directory, then run A-K1 and A-K2 only
   (local, $0). These prove the load path before any spend.
3. A-K3 and A-K4 need a 5090 — provision only after step 2 passes.

## 9. Provenance

- Goal-state assessment: `design/zone_a/GOAL-STATE-ASSESSMENT.html` (commit `2737800`)
- AAII v4.3 composition: `henri-research/references/aaii-v43-composite-and-exposure-audit.md`
- Backbone gates: `HENRI V2/henri_backbone_adapter.py` (`edb07ee`, `a74f507`)
- Transfer evidence: `design/zone_a/evidence/zonec_transfer_R1.json`
- Calibration: `design/zone_a/evidence/transfer_calibration_R1_R2_R3.json`

No benchmark score is claimed. No SOTA claim. No AAII v4.3 result exists for HENRI on
any member. This document is a specification, not a measurement.
