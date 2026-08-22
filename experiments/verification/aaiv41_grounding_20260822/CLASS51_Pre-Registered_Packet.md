# CLASS51 Pre-Registered Packet — BACKBONE INTEGRATION (DRAFT, APPROVAL-GATED)

- **packet_id:** `HENRI-PACKET-CLASS51-BACKBONE-INTEGRATION-2026`
- **Status:** DRAFT — NOT approved, NOT executed
- **Baseline commit:** `bae02f9` (branch `accuracy/fidelity-remediation`)
- **Date:** 2026-08-22
- **Authorizing context:** user clearance for implementations/decisions toward AAII v4.1 SOTA; skill `henri-agent-integration`; skill `henri-research`; evidence: `aaiv41_official_suite_registry.md`, `aaiv41_capability_gap_matrix.md`

## 1. Premise (evidence-backed)

AAII SOTA requires: (a) broad learned semantic knowledge (HLE, GPQA, AA-Omniscience), (b) calibrated open-answer generation, (c) long-context reasoning, (d) agentic tool use, (e) code synthesis at scale. Measured HENRI-only capability: MMLU 25.98% (chance+0.98%), GPQA 29.8% (< gate), HumanEval 2/50, MBPP 17/500. Ranking/codec/head levers FALSIFIED as semantic sources (run19/20, Phase 8.39). No pretrained backbone exists in the codebase (OBSERVED). Conclusion: a native zero-pretrained system is not on a credible path to AAII SOTA; the fastest credible route is backbone + HENRI layers.

## 2. Design amendment (REQUIRES USER APPROVAL — changes the zero-pretraining invariant interpretation)

**Proposed amendment to the zero-benchmark-pretraining invariant:**
- **Still prohibited:** training or tuning on any benchmark-family evaluation data; benchmark contamination; test-set leakage into HENRI stores; ARC-AGI-3 data into any store or model (ARC-AGI-3 remains a pure zero-pretrain demonstration, and the existing ARC path stays untouched/default).
- **Permitted:** a provenance-audited general foundation backbone (weights lineage + SHA-256, license, disclosed training data or contamination analysis where available), used as a frozen semantic encoder/decoder.
- **HENRI's role (unchanged philosophy):** Zone C axioms/memory, online test-time compilation from demonstration pairs, planner/verifier loops, SGLD adaptation, Sagnac/EFE governance — all remain zero-pretrain, online, and applied ON TOP of the backbone.
- **Ablation rule:** every HENRI augmentation is paired against the unchanged backbone baseline; a component that reduces aggregate score is reverted.

## 3. Phases (pre-registered gates)

| Phase | Scope | Gate / kill criterion |
|---|---|---|
| P0 | Official-suite grounding (registry + gap matrix) | **DONE — this packet's evidence** |
| P1 | Backbone integration: exact model + tokenizer + image processor + chat template; deterministic inference; no HENRI augmentation; default-OFF flag `HENRI_BACKBONE=1`; provenance gate (SHA + license + lineage) before `score_eligible`; provisioning on Vast (pip install transformers/accelerate, disk plan) | KILL: provenance audit fails; load cannot be reproduced; eligibility granted without provenance |
| P2 | Baseline reproduction on public subsets: GPQA Diamond (198), Global-MMLU-Lite slice, HLE slice, SciCode subset | KILL: backbone baseline not reproducible within tolerance; any evaluator shortcut |
| P3 | HENRI augmentation, independently ablated: (a) Zone C retrieval pre-prompt; (b) verifier loops (sandbox/REPL for code); (c) planner/tool loop (agent harness for GDPval-class); (d) SGLD online adaptation | KILL per component: no improvement over backbone-only on the same subset; latency/context violation |
| P4 | Full public suite (all reproducible index components) | KILL: contamination detected; private items replaced with synthetic approximations |
| P5 | Official AA submission | only if the official process permits; otherwise local scores labeled DERIVED, AA-published scores OBSERVED with submission evidence |

## 4. Standing kill criteria (all phases)

1. Backbone baseline cannot be reproduced on any public subset → KILL.
2. Any benchmark-family data enters training, tuning, or Zone C stores → KILL + seal (zero-tolerance).
3. Generic decoder or uncalibrated head grants score eligibility → KILL (fail-closed preserved).
4. Private/unavailable benchmark items replaced by synthetic/template items → BLOCKED, no score claim.
5. HENRI augmentation reduces aggregate score below backbone-only → revert that component.
6. VLA claims made from text-only benchmark performance → prohibited (no overclaim).

## 5. Evidence discipline

- Local reproduction scores = DERIVED (AA runs components independently; only AA-published scores are OBSERVED for the index).
- Every claim: source URL/bytes/SHA, evaluator identity/version, item-level results, commit, environment.
- RT-MCTS stays default-OFF; main untouched; promotion via the standing release-convergence gate.

## 6. Open decisions for approval

1. Adopt the design amendment (§2)?
2. Backbone model choice: **Qwen3-VL-8B-Instruct** (default proposal; multimodal, 32 GB VRAM fit on RTX 5090, permissive license) vs larger (32B/72B — needs quantization) vs text-only (Qwen3-8B — no vision) vs user-specified.
3. P1+P2 execution authorization (remote provisioning included)?
