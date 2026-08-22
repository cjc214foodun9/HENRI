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

## Amendment A1 (RATIFIED 2026-08-22) — contamination detector calibration
- Status: RATIFIED (user via Photon; bounded claim: "no code-dominant syntactic shingle overlap detected between the 13 corpus files (sha256 aggregate b20b5144…) and the first 30 HumanEval tasks under heuristic v3.1"; does NOT assert absolute text isolation). Corpus composition UNCHANGED (same 13 files, same aggregate sha256 `b20b5144…`).
- Ratification receipt: `class51_p3_a1_ratification_receipt.json` (sealed, committed `8ebd222`).
- Original pre-registered gate (text above, unmodified): 5-gram shingle overlap of benchmark task prompts/solutions/tests vs corpus.
- Observed blocks and classification (evidence scripts `classify_contamination.py`, `classify_contamination_v2.py`, `show_contamination_overlap.py`):
  - v1 (plain 5-grams): blocked 7/13 files — all vacuous (`[2, 2, 2]`, `1 2 3 4 5`, `a b c d e`, prose). Single verbatim-line overlap = doctest OUTPUT literal `[2, 2, 2]` in itertools.rst, benign.
  - v2 (identifier 4-grams, len>=3): blocked `re.rst` — prose shingle `the end the string` (HumanEval/10 vs re.rst prose), not code.
- Amended rule (v3.1, defined BEFORE any rerun result): contamination fires only on
  (a) a normalized corpus line containing >=1 identifier (len>=3) AND (underscore token | code-dominant keyword `def/class/import/from/return/lambda/yield/assert/raise/except/finally/while/elif/global/nonlocal/pass/break/continue/del` | `>>>` | `=`), OR
  (b) an identifier-4-gram shingle (len>=3) containing an underscore token or code-dominant keyword.
  Prose-common keywords (in/for/as/or/if/not/print/range/len/…) and generic punctuation are NOT signals.
- Receipts: every run writes a timestamped `contamination_receipt_<run_id>.json` carrying detector version + source commit; historical blocked receipts are preserved (never overwritten).
- Preflight: contamination-only remote check (no model load) records `hits` + PASS/BLOCKED before any GPU run.

## Pilot outcome (RATIFIED A1, 30-item plumbing, 2026-08-22)
- Run 1 (commit `cba20be`): 0/30 both arms — INVALID, harness bug: canonical HumanEval `test` is a STRING; loader `"\n".join(obj["test"])` char-split it → uniform `SyntaxError: unterminated string literal`. Kill evidence: `kill_test_loader_join.py` (HE/0 + HE/2 PASS on fixed path; HE/1 genuine NameError). Receipts archived `_INVALID_0of30_*`.
- Fix: `0052864` — loader + preflight only (arms/prompts/gates/model unchanged).
- Run 2 (commit `0052864`, run_id `20260822T093948Z`): Arm A **15/30 (0.5)**, Arm B **18/30 (0.6)**; all plumbing gates PASS (engagement 30/30, contamination CLEAN, execution errors 0/0, catastrophic regression 0.0, latency within budget); frozen backbone confirmed.
- Verdict: PLUMBING-ONLY PASS. Efficacy NOT evaluated (kill criterion `accuracy_B − accuracy_A < 0.010` requires the full 264-item matrix).

## Open ratification items (user decision)
1. Approve this P3(a) operationalization (recommended), OR switch to P3(b) verifier loop, OR run both sequentially.
2. Optional Arm C (literal Zone C wave retrieval control) — NOT recommended (evidence above); include only on request.
3. **Ratify Amendment A1** (v3.1 detector) to unblock the plumbing pilot rerun. Rejecting it keeps the original 5-gram gate → pilot remains CONTAMINATION_BLOCKED by construction (vacuous hits).
