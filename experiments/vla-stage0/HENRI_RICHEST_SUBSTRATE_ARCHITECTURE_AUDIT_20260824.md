# HENRI Richest-Substrate Architecture Audit — 2026-08-24

**Program:** identify the richest architecture justified by current evidence; implement and test its smallest causally decisive carrier.
**Reference 3 (gpt-5.6-sol) binding.** "Perfect architecture" is not falsifiable; this record is the executable interpretation.

## 1. Authenticated review inventory
| Artifact | Status | Evidence |
|---|---|---|
| `HENRI_Inbox/HENRI_SOTA_VLA_Reverse-Engineered_Roadmap.md` | READ IN FULL | 11,789 B, plain markdown, SHA-256 `6c534e93fe4df9297fb98f92…` (Aletheia, reverse-teleology Level 5→1, Path B2 Step 1) |
| `Project HENRI Reverse Engineering Digest.gdoc` etc. (4 gdocs) | BLOCKED | 183 B Drive stubs; `gws` CLI not installed — not readable as files |
| Stage-0c-rev2/3/4 uploads | READ (this session) | `99b3b828…`, `99d28342…`, `458d0776…` |

## 2. Live architecture graph (OBSERVED, branch `feature/v0.4-token-fsa-lineage` HEAD `d8404f1`)
- **Representation core:** D=65,536 qFHRR Z_256 phase rings; [8192,8] Cl(3,0); S^(D-1) sphere; Cholesky retractions; Sagnac veto Δ≤0.35.
- **Semantic backbone:** `QwenBackboneAdapter` (frozen Qwen3-VL-8B-Instruct, pinned revision `0c351dd0`, `HENRI_BACKBONE=1`, shard-hash provenance, default-OFF without flag). Branch `accuracy/aaiv41-matrix264` carries the pinned backbone.
- **Decoder/egress:** `HENRINeuralEgressUnbinder` (down_proj 65536→2048 + lm_head 2048→32000), `adapt_in_context_sgld_wave` (C2 soft-target protocol), `HENRIUnifiedEgressTransducer` (checkpoint-gated), `PhaseRingCodebookDecoder`, `ASTProductionPhaseCodec`, `henri_egress` (Text/Tool/Universal), `arc_egress_contract`.
- **Structural path:** `qfhrr_ast_discriminative_kernel` (ASTDiscriminativeEncoder, IDF), CEGIS (experiments/exploratory), sandbox (`exteroceptive_sandbox`, `henri_gpu_crystalline_active_inference`).
- **Memory:** `ContinuousHopfieldCleanup` (β=8.0), `HopfieldActionDecoder`, `zone_c_retrieval_bridge`, `BackboneRetrieval`, `zone_c_segment_cache`; Zone C PostgreSQL+pgvector+TimescaleDB, 11 boundary axioms, 10,703 engrams (Vast 47411800).
- **Dynamics/policy:** `EFEPlanner` (live policy `select_action`), `LowRankCoupledTransition` (VW†+R_block), `RecursiveDualEDMD`, `WaveJEPA` (NOT in live loop), `darwinian_phase_swarm`, thermostat, planner.
- **Ingress:** `henri_vision_encoder`, `connected_component_segmenter`, `phase_codec_adapter`; O-VSA FALSIFIED (2026-08-21).

## 3. Roadmap claim-by-claim disposition
| Roadmap claim | Disposition |
|---|---|
| Step 1: Path B2 hard-negative codec → Gate A (ranks ≤5/71, margin ≥0.25) | **CONFLICTS_WITH_LIVE_CODE / FALSIFIED** — Path B2 measured 2026-08-21: isometry 1.37e-06 + non-collapse 0.9583 + rank 5/13 PASS, **margin −0.02/−0.03 FAIL**. Codec-level ranking alone cannot close the gap. |
| Step 2: `sagnac_mcts_cuda_core.cu` latency ≤2 ms + dual Zone C schema | `BOUNDED_IMPLEMENTABLE` but NOT causal to AAII until a semantic carrier exists; latency is not capability. |
| Step 3: internet pre-training (Kinetics-700, Ego4D, CodeSearchNet, OpenWebMath) | `BLOCKED_MISSING_PREMISE` — conflicts with the zero-pretraining invariant; Zone C seeded axioms are the frozen reference; task compilation is 100% online at test time. |
| Step 4: AAII v4.1 targets (HumanEval/MBPP/GPQA/MMLU-Pro/ARC-AGI-3) | `FALSIFIED` as composition — pinned v4.1.1 bytes: GDPval-AA v2, Terminal-Bench v2.1, τ³-Banking, AA-Omniscience, HLE, SciCode, AA-LCR, GPQA Diamond, CritPt. Only GPQA Diamond overlaps. |
| MI floor 0.04 bit; I(Y)≥0.85 bits; 3.5B/12.5 GB active | `HYPOTHESIS` — no live production-scale MI or 3.5B trained core exists. |

## 4. Capability-gap matrices
**AAII v4.1.1 (0/9):** every component needs (a) semantic text/agent backbone with long context + tool loop (Terminal-Bench v2.1, SciCode, τ³-Banking), (b) calibrated structural egress (SciCode, Terminal-Bench), (c) domain reasoning (HLE, GPQA Diamond, CritPt, AA-Omniscience, AA-LCR, GDPval-AA v2). Reproducibility: private datasets BLOCKED; only operator-published scores OBSERVED. **No causal carrier exists for any component.**

**VLA (0/12):** Stage-0a wrapper VERIFIED; 0b frozen encoder VERIFIED; 0c dynamics BLOCKED ×4 (rev `IDENTIFIABILITY_BLOCKED`; rev2/rev3/rev4 `CONTRACT_FAILED`). Perception → temporal state → policy → environment loop → continual memory all unestablished.

## 5. Measured architecture laws (binding)
1. Selection resolution ~1/candidates (run12). Bigger pools are NOT the answer.
2. MBPP bottleneck = grammar coverage + structural decode (run17: 99.4% COVERAGE_MISS; run18 17/500; run15 escalation-only gains; run16 penalty FALSIFIED).
3. qFHRR `encode_text` is a random-ring codec — W_task from it is random-delta superposition (run20 codec control). Structured codec fixed continuity but not task occlusion (run21 FALSIFIED_AT_SCALE).
4. Path B2 margin FAIL → codec-only discriminative ranking is insufficient.
5. R-EDMD: relative one-step skill real (SSR 0.369/0.516 ≤ gates), absolute calibration and rollout stability absent across r∈{8,16}.
6. Internal coherence (Sagnac, entropy, Koopman fit) never substitutes for external outcome.
7. Zero-pretraining invariant: no SGD pretraining of the semantic core.

## 6. Candidate substrate comparison (10 criteria; 1–5, 5=best)
| Criterion | Trained/frozen backbone + AST decoder + CEGIS (A) | Action-cond. latent world model (B) | Multimodal perception + temporal state (C) | Hierarchical policy/tool controller (D) | Episodic/semantic memory + retrieval (E) | Verifier/env-feedback channel (F) |
|---|---|---|---|---|---|---|
| Representational compatibility | 5 (Qwen3-VL frozen + wave boundary) | 3 (wave-only, Stage-0) | 3 (vision encoder exists) | 4 (arc_egress vocab) | 4 (Hopfield/Zone C live) | 5 (CEGIS/sandbox live) |
| Semantic grounding | 5 (real LLM semantics) | 2 (codec random-ring) | 2 | 3 | 3 | 4 |
| Temporal/action conditioning | 3 | 4 (EDMD K_a) | 2 | 3 | 2 | 3 |
| Trainability/data needs | 4 (frozen; adapter only) | 2 (identifiability failed) | 3 | 3 | 3 | 4 |
| Multi-step stability | 4 | 1 (rollout 0.52–0.56, ρ>1 transient) | 2 | 3 | 3 | 4 |
| Structural egress | 5 (AST + CEGIS) | 1 (linear head only) | 1 | 4 | 3 | 5 |
| Tool use / long context | 5 | 1 | 1 | 4 | 3 | 4 |
| AAII causal relevance | 4 (Coding/Agent closest) | 1 (CartPole ≠ AAII) | 2 | 3 | 2 | 4 |
| Compute/memory cost | 3 (12–25 GB class) | 5 (CPU-scale) | 3 | 4 | 4 | 4 |
| Falsifiability | 5 (pre-registered gates) | 4 | 4 | 4 | 4 | 5 |
| **TOTAL** | **43** | **24** | **23** | **35** | **30** | **42** |

**Selection: A** — frozen semantic backbone (Qwen3-VL-8B) → structural AST decoder → CEGIS verify. Wave/latent dynamics (B) and memory (E) remain DIAGNOSTIC sidecars, not task-critical. Assumptions: (1) frozen backbone preserves the zero-pretraining invariant (no SGD on backbone weights; adapter-level only); (2) AAII Coding/Agent components are the closest measurable target; (3) the 13-rule DSL + 52/52-heldout grammar remains the candidate space.

## 7. Smallest decisive carrier (proposal — REQUIRES APPROVAL, not yet authorized)
**Egress-1: frozen semantic backbone → structural AST egress → CEGIS → MBPP-style pass@1.**
- Ingress: task text → frozen Qwen3-VL-8B embeddings (flag `HENRI_BACKBONE=1`, provenance-pinned `0c351dd0`) vs wave-encoder control arm.
- Candidates: structural AST decoder over the 13-rule DSL; exact-arity dispatch (run17 fix); semantic closure per family (run18 protocol).
- Egress: CEGIS verifier + disjoint outcome tests; sandbox-call cost counted; escalation allowed at recorded cost.
- Baseline: run18 17/500 (matched verifier budget). Verdicts: `NO_EFFECT` / `SUPPORT_RESTORED` / `CAPABILITY_PROMOTED`; kill `FALSIFIED_NO_EXTERNAL_GAIN` (no external pass gain with matched budget).
- Fresh single-use heldout REQUIRED (all prior splits consumed); pre-registered per-family support ≥1 canonical pass and pass@1.
- This is the first carrier; world-model/memory/policy carriers follow only on gate passage.

## 8. Governance
Authenticated review → premise-audit event → decision event → sealed carrier contract → implementation → disposable smoke → fresh eval → outcome event → isolated commit + native-path bundle.

## Boundaries
- **VLA 0/12; AAII v4.1.1 0/9 BLOCKED.** No SOTA claim. CartPole spectral line closed at rev4 `CONTRACT_FAILED` (commit `d8404f1`, r28 `6c1279c5…`).
- Stage-0c-rev4 final: C9 FAIL (SSR_eval 0.423 > 0.40; upload's 0.3688 was r=16 — FALSIFIED at r=8); C10 PASS (SSR_rollout5 0.516); contraction never fired (ρ 0.949/0.927); tele `ed162c85…`, outcome `181bc5b5…`.
