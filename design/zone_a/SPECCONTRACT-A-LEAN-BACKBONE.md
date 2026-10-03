# SpecContract A — lean backbone for a 32 GB RTX 5090 (PROPOSED)

Status: **PROPOSED**. Research holon output. Consumes architecture and integration.
Not approved. No implementation is authorised by this document.
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
| A-K4 | Emits distinct top-1 tokens above floor on ≥100 prompts | the recorded egress defect: 16/16 chunks gave the same top-1 token |
| A-K5 | Matched ablation: Zone C conditioning ON vs OFF | no gain ⇒ memory layer does not transfer at this scale |

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
