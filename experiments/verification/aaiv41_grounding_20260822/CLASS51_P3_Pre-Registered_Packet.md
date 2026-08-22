# CLASS51 P3 Pre-Registered Packet — P3(a) RETRIEVAL PRE-PROMPT (PROPOSED)

- **packet_id:** `HENRI-PACKET-CLASS51-P3-RETRIEVAL-2026`
- **Status:** PROPOSED — operationalization of approved P3(a); user ratification pending
- **Authoritative scope (approved):** CLASS51 packet §3 P3(a) "Zone C retrieval pre-prompt", kill: "no improvement over backbone-only on the same subset; latency/context violation"
- **Branch:** `accuracy/aaiv41-backbone`; baseline `f952db0` immutable; main untouched; all flags default-OFF

## Mechanism mapping (proposed amendment to the letter of P3(a))
- "Zone C retrieval pre-prompt" is operationalized as **read-only System-3 memory retrieval** from a provenance-clean public Python-API corpus, injected into the prompt before frozen-backbone generation.
- Literal Zone C wave retrieval is **excluded with evidence**: (1) Zone C holds 10,826 ARC engrams + axioms — non-code semantics; (2) qFHRR text codec FALSIFIED as semantic (run20, random-ring baseline ~0.0039); (3) corpus INFERRED warning: ANN self-matching induces coherent solipsism (R≈0.95 internal resonance, zero task grounding).
- Deterministic BM25-style lexical index = the honest memory probe for code tasks at this step. No learned embeddings, no wave codec, no Zone C reads/writes.

## Corpus (immutable)
- Source: CPython `Doc/library/` (PSF license), pinned to commit **`f74cdf80a120649e4c353430da8cbd1305c00993`** (2026-08-22).
- 13 modules: bisect, collections, functools, heapq, itertools, json, math, os.path, pathlib, random, re, statistics, string.
- Per-file SHA-256 on LF-normalized bytes + aggregate manifest hash in `data/backbone_retrieval_corpus/manifest.json` (`henri.corpus-manifest.v1`).
- Construction script committed; retrieval date recorded.

## Contamination gate (deterministic, required before any GPU run)
- Scan every corpus snippet for 5–9-gram shingles of: HumanEval/MBPP task prompts, reference solutions, docstrings, unit tests, normalized function signatures, and near-duplicates (hashed shingle overlap).
- Gate fires → corpus rebuild or replacement; a fired gate blocks the run (`CONTAMINATION_BLOCKED`).
- Contamination receipt (`henri.contamination-receipt.v1`) committed before execution.

## Arms (matched, sequential exclusive GPU)
| | Arm A | Arm B |
|---|---|---|
| backbone | Qwen3-VL-8B-Instruct @ 0c351dd0, frozen | identical |
| decoding | greedy, max_new_tokens 384, do_sample=false | identical |
| prompt | task + HumanEval/MBPP prompt text | identical + retrieval block |
| prompt identity | per-item SHA-256 recorded | per-item SHA-256 recorded; byte-identical except retrieval block |
| evaluator | deterministic unit-test sandbox (15 s timeout) | identical |
| GPU | RTX 5090, exclusive | identical, sequential |

## Diagnostic sets
- HumanEval full (164 items; canonical gz `b796127e…` / decompressed `1d49078b…`)
- MBPP sanitized first 100 (canonical `google-research/google-research` `mbpp/sanitized-mbpp.jsonl`, pinned commit + digest; verified before use)
- Total 264 items/arm; diagnostic only, `not_official_aaii=true`, `held_out_status=CONDITIONAL`.

## Pre-registered gates
1. **Plumbing pilot (30 items, Arm A then Arm B):** engagement ≥ 90% (retrieval hits + prompt delta), contamination receipt clean, retrieval latency ≤ 2 s/item, execution errors = 0, no catastrophic regression (> 3 pt drop). Pilot yields NO efficacy verdict (1 item = 3.3 pt).
2. **Efficacy (full 264/arm):** kill if `accuracy_B − accuracy_A < 0.010` (one pre-registered criterion). Report McNemar exact test (one-sided, item-level paired) + bootstrap CI; significance is reported, the point-gain threshold is the kill.
3. **Latency:** per-item median/p95, normalized per output token; retrieval time reported separately; kill if total latency > 1.5× Arm A at equal tokens.
4. **Causal-path telemetry:** every item records retrieval hits, prompt SHA-256 delta vs Arm A, provenance tags (source file + snippet sha256). Missing telemetry = `INVALID_EVIDENCE`.
5. Fail closed: corpus missing, manifest mismatch, contamination hit, retrieval error → `RETRIEVAL_BLOCKED` abort. No silent fallback to Arm A.

## Evidence discipline
- `henri.run-evidence.v1` receipts per arm with item-level outcomes + raw log hashes.
- Contract tests: default-OFF identity, frozen params, contamination rejection, provenance tags, fail-closed paths, arm matching.
- Results labeled diagnostic — not official AAII v4.1.

## Evidence addendum (2026-08-22)
- Corpus staged: 13 modules, 547,851 LF bytes, aggregate sha256 `b20b5144adeea0dc23fb02e258a735af6849e414f52275e53832bc1a34717aac`; manifest `data/backbone_retrieval_corpus/manifest.json` (`henri.corpus-manifest.v1`); staging script `stage_retrieval_corpus.py`.
- MBPP canonical confirmed: `google-research/google-research/mbpp/sanitized-mbpp.json` (blob sha `a999d25d…`, 427 items, keys code/prompt/source_file/task_id/test_imports/test_list); local bytes sha256 `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9` (not committed; digest recorded).
- HumanEval full: 164 items, canonical gz sha256 `b796127e…` / decompressed `1d49078b…` (MATCH_LOCAL).

## Open ratification items (user decision)
1. Approve this P3(a) operationalization (recommended), OR switch to P3(b) verifier loop, OR run both sequentially.
2. Optional Arm C (literal Zone C wave retrieval control) — NOT recommended (evidence above); include only on request.
