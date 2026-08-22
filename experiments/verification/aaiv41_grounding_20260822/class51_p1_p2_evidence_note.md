# CLASS51 P1+P2 Evidence Note (2026-08-22)

## Status
- P1 (backbone integration): **VERIFIED on CUDA target**
- P2 (diagnostic baseline): **HumanEval public diagnostic run** (see receipt); GPQA Diamond / MMLU-Pro / SimpleQA **BLOCKED** (HF access-gated, HTTP 401, no token configured)

## P1 evidence (OBSERVED)
- Model: `Qwen/Qwen3-VL-8B-Instruct`, immutable revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Artifact manifest: `qwen3vl8b_tree_manifest.json` (4 LFS shards, per-shard SHA-256; total 17,534,339,512 bytes)
- Shard hash verification on target: PASS (manifest sha `e171408625853d52e038bc97af6df4375153f09531505b716c8fc77d9d5a6916`)
- Load: bfloat16, cuda:0, `checkpoint_load_status=LOADED`, total params 8,767,123,696, **trainable 0 (frozen)**
- Text smoke: "What is 84 * 3 / 2?" -> `126` (correct)
- Image-text smoke: red square w/ green inner -> "A red square contains a smaller green square centered within it..." (correct description)
- Memory: 16,754 MiB allocated / 16,798 MiB reserved (within 32 GB, no offload observed)
- Receipt: `backbone_smoke_receipt.json` (`henri.class51-smoke.v1`, status PASS, 14.19 s)
- Contract tests: 13 passed (default-OFF gate, provenance, freeze, shard SHA-256, CLI fail-closed) — commit `fe23f9f`
- Implementation: `HENRI V2/henri_backbone_adapter.py` (default-OFF via `HENRI_BACKBONE=1`), committed on `accuracy/aaiv41-backbone`

## P2 evidence
- Canonical HumanEval (openai/human-eval): gz sha256 `b796127e...`, decompressed `1d49078b...`, 164 items, MATCH_LOCAL
- Runner: `backbone_humaneval_diagnostic.py` (greedy, deterministic unit-test grading, 15 s timeout sandbox)
- **Result (OBSERVED, CUDA): 19/20 passed, 0 execution errors, accuracy 0.95, elapsed 24.67 s**; receipt `class51_p2_humaneval_receipt.json` sha256 `d95c0d8a...`; item reconciliation passed+failed==attempted (20/20); frozen (trainable 0); 16,754 MiB allocated
- Receipt schema: `henri.run-evidence.v1`, kind=diagnostic-baseline, not_official_aaii=true, held_out_status=CONDITIONAL
- Blocked official components: GPQA Diamond (`Idavidrein/gpqa`), MMLU-Pro (`Idavidrein/mmlu_pro`), SimpleQA (`openai/simpleqa`) — all HTTP 401 access-restricted

## Corpus consult (INFERRED)
NotebookLM bank `ca4bb787` supports the integration: frozen VLM as System-1 perceptual ingress; HENRI D=65,536 wave core as System-2 (planning, Sagnac veto, lookahead); symbolic egress + unified memory as System-3. Stated caution: the backbone is a sub-symbolic prior; task semantics must still come through HENRI layers — a backbone alone does not satisfy HENRI's active-inference contract.

## Governance
- Approval: `henri.class51-approval.v1` (commit `2af4f37`)
- Grounding: `d8d2c86`; implementation base: `b1d901b` (branch `accuracy/aaiv41-backbone`)
- Main untouched; RT-MCTS default-OFF; no P3–P5; no official composite claims
