# Phase 8.31 — VLA Egress Docs Ingest + Matched A/B Design (2026-08-19)

## 1. Sealed documents (OBSERVED)
- VLA PDF: sha256 `b7536f927283290f883d2e07092b9e3953f210dcb33edd981b4a21822eb0e70c` (123826 B)
- Textegress.txt: sha256 `c5c7a02e3e2769139509604127f547e4af5143b3db80e0d2b9a85a80087344b7` (11713 B)
- C++harness.txt: sha256 `46f3ef808660b9901477782e7374cd32e476bece2d23e309f95998611d8fc4ca` (8558 B)
- VLA PDF extracted: 4 pp, 11,209 chars -> `C:\tmp\vla_egress_arch.txt`
- Textegress.txt: 11,713 chars -> `C:\tmp\Textegress.txt` (scaffold `henri_discrete_egress.py`)
- C++harness.txt: 8,558 chars -> `C:\tmp\C++harness.txt` (scaffold `henri_cuda_harness.hpp`)

## 2. Requirements-to-live-code matrix
| Doc element | Live code | Status |
|---|---|---|
| VLA §I Rx: UWE ingress (vis/text/act superposition) | henri_vision_encoder.py HENRIVisionEncoder + qFHRREpistemicCodec | EXISTS |
| VLA §II Zone B: W_task Ψ_t, Sagnac veto, Langevin | efe_planner / sagnac_mcts / wave_jepa | EXISTS |
| VLA §III Tx: HENRIDiscreteEgressPipeline (down-proj→LM head→sample) | henri_decoder.py HENRIUnifiedEgressTransducer + PhaseRingCodebookDecoder + adapt_in_context_sgld | EXISTS (scaffold names differ: w_down/layer_norm/w_lm == down_proj/layer_norm/lm_head) |
| Textegress PhaseCodecAdapter | henri_decoder PhaseRingCodebookDecoder (Z_256) | EXISTS |
| Textegress ClosedLoopSGLDAdaptor | henri_decoder.py:133 adapt_in_context_sgld | EXISTS |
| Textegress HENRIDiscreteEgressPipeline forward | henri_egress.py TextEgress/UniversalEgress | EXISTS |
| C++ harness: CUDA graph capture, ring buffer, zero-alloc | henri_fused_triton_cuda_graph_runner.py (Triton) | PARTIAL — C++ = proposed accelerator, NOT official-env replacement; byte-equivalence required for publishable scores |
| VLA §3 metrics (40 us/cycle, r>=0.82, L<=0.15 in 5 steps) | TARGET_GOAL projections | NOT evidence |

## 3. Run5 baseline LOCK (20 selected / 19 attempted / 1 blocked / 2 scored / total 5.9524)
- Commit `5005410`; log `2ba94372…`; scorecard `514320e5…`; verdict `5c0aa426…`; jsonl `16a1d885…`
- sp80 4.7619/1/20 (beats baseline 39) | cn04 1.1905/1/158 (baseline 29, worse) | 17 zero | ls20 BLOCKED_NETWORK (not zero)

## 4. No vacuous Run6
- `7a9cd9f` = additive default-OFF calibrator, NO production action-path wiring.
- ARC exposes NO authorized (observation, GameAction, data) trajectories -> `BLOCKED_NO_ACTION_TRAJECTORIES`.
- Head state until artifact passes G1-G9: trained_action_head_active=false, score_eligible=false, diagnostic_only=true.
- Run6 (A/B) launches ONLY after a legitimate calibration artifact + production wiring gate.

## 5. Matched A/B design (pre-registered)
- Pin Run5 ordered env list (sp80,ka59,cn04,s5i5,vc33,bp35,sb26,m0r0,g50t,su15,lf52,ar25,tu93,lp85,sk48,tr87,wa30,sc25,tn36,ls20) in immutable manifest; pin seeds/budgets/flags/checkpoint sha/evaluator version.
- Arm A: frozen EFE control. Arm B: algebraic head (post-gates).
- Paired-cell rule: ls20-type network failure in either arm -> classify paired cell BLOCKED_INFRASTRUCTURE, never impute zero; compare preregistered intersection.

## 6. Merge boundary (strict)
- NO FF of 7a9cd9f to main on synthetic tests alone.
- Required: focused CUDA pass + full remote suite pass + profiler receipt + default-OFF identity + strict artifact validation + non-vacuous mechanism engagement + matched A/B non-regression.
- Default-OFF infra merge MAY be considered separately, never described as calibrated head or SOTA.

## 7. Profiler targets (OBSERVED run5)
- ~16-18 s/step interactive; 57 fail-closed linalg.eigh non-convergences (reproduce first, then bounded cascade fix, additive).
