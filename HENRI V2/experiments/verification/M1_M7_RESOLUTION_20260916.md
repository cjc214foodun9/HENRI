# M1–M7 RESOLUTION — what was built, what was measured, what stays blocked

**Date:** 2026-09-16 | **Branch:** `carrier/aaii-v43` | **Base:** `7750abb` (`origin/main` untouched)
**Spec:** `SPEC-2026-09-16-AAII-V43-RD-PIPELINE.json` (Contract A)
**Device:** CPU only (`torch 2.11.0+cu128`, `cuda_available=false`) — no CUDA claim anywhere.

---

## 0. Deliverable summary

| Gap | Status | Artifact | Evidence |
|---|---|---|---|
| **M1** egress | **ROOT CAUSE LOCALIZED** (`INGRESS_CODEC_DEFECT`) | `kaa_egress_probe.py`, `kaa_a1_nearfar_probe.py`, `kaa_a1_multi_ingress.py`, `KAA_AMENDMENT_A1_nearfar_egress.md` | W margin +0.772 vs +0.001, same head/pairs |
| **M2** agent loop | **BUILT** | `henri_agent_loop.py` + 600-line test file | turn budget, guardrail zeroing, abstention, fsync'd per turn |
| **M3** sandbox | **BUILT** (surrogate measured) | `henri_sandbox_harness.py` + tests | `container-rlimit` ran; namespace probe reported unavailable |
| **M4** PDF ingress | **BUILT** | `henri_pdf_ingress.py` + 716-line test file | pymupdf backend; page provenance; fail-closed |
| **M5** world knowledge | **`REQUIRES_APPROVAL` — NOT TOUCHED** | — | contract change; no backbone enabled |
| **M6** CUDA target | **`BLOCKED`** | — | Vast credit 0, instance exited |
| **M7** eval runner | **BUILT** | `henri_eval_runner.py` + 736-line test file | fail-closed ordered gate chain |
| **S0** evidence infra | **BUILT** | `henri_eval_infra.py` + 42 tests | ledger, unique paths, reconciliation, K-D scanner |
| **S1** SciCode | **DATASET PINNED + STAGED** | `data/official_benchmarks/scicode/manifest.json` | rev `4510f6a6…`, gated `False`, apache-2.0 |

Worktree suite at the deliverables commit: **1778 passed, 21 skipped, 0 failed**.
Push gate: **GATE 1 PASS** (149/149 collectible), **GATE 2 PASS** (505 files, 0 untracked-module edges),
**GATE 3 PASS** (all new modules import from the committed tree), **GATE 4 PASS** (191 tests collect).

---

## 1. M1 — the egress root cause (the key finding)

### 1.1 The inherited verdict

`aaii_v42_testtime_sufficiency_audit.md` recorded `top1_token_unique = 1` across 16
distinct chunk waves and labeled the egress `BLOCKED_SEMANTIC_CAPACITY`. The audit
carried that forward as M1, the largest single gap (40% of index weight). Both
readings blamed the **egress head**.

### 1.2 K-A was VACUOUS — withdrawn

Probe `kaa_egress_probe.py`, 512 item rows, checkpoint `LOADED`
(`75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060`).

| Arm | distinct top-1 (of 128) |
|---|---|
| A `ring_uint8_scaled` | 39 |
| B `transduce_real` | 19 |
| **C `random_control`** | **37** |
| D `shuffle_control` | 19 |

**Two defects, both mine, both disclosed:**

1. **The gate was vacuous.** The pre-registered criterion was "distinct ≤ 1 →
   `FALSIFIED_NO_EGRESS`". The **negative control** (pure `torch.randn`) scored 37,
   statistically indistinguishable from the best content arm (39). A criterion its
   own control passes detects nothing.
2. **`shuffle_control` was vacuous by construction.** Permuting the *same* prompt
   set cannot change the *set* of top-1 tokens.

**Why:** `distinct_top1` counts *how many* tokens appear, not *whether the token
depends on the input*. At ~10 nats of entropy over 32,000 ways, a content-blind head
still yields many distinct argmax values. Distinctness is a false-positive generator
for this mechanism.

### 1.3 Amendment A1 — differential response at two boundaries

Pre-registered in `KAA_AMENDMENT_A1_nearfar_egress.md` **before** execution. Near
pair = same frame, one content word changed; far pair = different frame and topic.
Bootstrap CI (2000 resamples), seed 20260916, n = 32 pairs. Matched noise control at
both boundaries. Determinism check.

**Boundary W (encoded wave = ingress):**

| Ingress | W margin | CI95 | Verdict |
|---|---|---|---|
| `T_universal_transducer` (canonical real) | **+0.771927** | [+0.77181, +0.77206] | **carries content** |
| `S_structured_charpos` | **+0.048210** | [+0.04779, +0.04866] | **carries content** |
| `Z_epistemic_ring_uint8` | +0.001369 | [−0.00105, +0.00397] | **loses content** |

Noise control W: +0.995153 (decisive). Noise control L: +0.453037 (decisive).
Determinism: `True`.

**Same head. Same pairs. Same seed. Only the ingress changed.**

### 1.4 Mechanism

`qFHRREpistemicCodec.encode_text` returns a `[65536] uint8` Z_256 ring built from
SHA-256-seeded `torch.randint` (measured: `min=0 max=255 unique=256`). Every distinct
string maps to an **independent random ring**; similarity ≈ `1/√D` for all distinct
pairs. A wave built this way carries no metric structure, so no downstream head can
recover a near/far ordering from it. This matches the repository's own recorded
property (structured-codec kill experiment, Run21).

**Consequence:** `BLOCKED_SEMANTIC_CAPACITY` is **FALSIFIED as a head-capacity
claim**. The defect is upstream of the head, in the codec. The bare linear head
discriminates once fed a structured wave, so the corpus-asserted necessity of the
full Calibrated Egress Gateway is **not required by this evidence** (it may still be
required for *token-string* quality — untested).

### 1.5 A2 — the next gate, and it is BLOCKED

A1 shows logit-level differential response exists. A2 asks whether those logits
become a **scored string**. That needs an id→string mapping.

Measured: `lm_head` is **32000**-way. Host tokenizer caches found: 8.

| Cache | vocab |
|---|---|
| EleutherAI/llemma_7b | 32019 |
| hf-internal-testing/llama-tokenizer | 32003 |
| gpt2 | 50258 |
| Qwen2.5-1.5B-Instruct | 151665 |
| MiniLM-L6-v2 | 30527 |
| google/gemma-2b | 256217 |
| google/gemma-4-26B-A4B-it | 262168 |
| Intel/gemma-4-26B-A4B-it-int4-AutoRound | 262168 |

**Caches with vocab exactly 32000: 0.** Repo-side: 0 tokenizer-API hits in live code
(`AutoTokenizer`/`tiktoken`/`id_to_token`/`convert_ids_to_tokens`), no `vocab.json`,
no `id_to_token` table; `o_vsa_ingress_tokenizer.py` has `count[id_to_token] = 0`
and `count[token_to_id] = 0`.

**`A2 = BLOCKED_MISSING_TOKENIZER`.** No artifact in this repo or on this host binds
a token id to a string for a 32000-way head. Borrowing another model's tokenizer would
be a **fabricated binding** — the ids are positionally arbitrary, so an unrelated
tokenizer yields plausible-looking strings with no relationship to the trained head.
That is a mock loop and is refused.

Open-answer members (AA-Omniscience, GDP.pdf, AA-LCR, HLE — **40% of index weight**)
therefore stay `BLOCKED_PENDING_A2`, and that block is a **data/contract gap, not a
code bug**.

---

## 2. M2 — multi-turn tool-using agent loop

`henri_agent_loop.py` (52,106 B), `tests/contract/test_agent_loop.py` (27,035 B).

- Typed per-turn record: `turn_index`, `action_type ∈ {tool_call, final_answer, abstain}`,
  `tool_name`, `tool_args`, `observation`, `elapsed_ms`, `tokens_used`.
- Turn budget with an **explicit exhaustion outcome** (never silent truncation).
- Typed **guardrail violation that zeroes the score** and halts (AutomationBench-AA gives
  zero credit on a guardrail violation).
- Tool dispatch via an **injected callable**; built-in file-write tool and a REST tool that
  is **disabled unless an explicit allowlist entry matches**.
- **Abstention** recorded as `ABSTAINED`, not `FAILED`.
- Per-turn telemetry appended with flush+fsync per row → a crash keeps prior turns.
- `≤16`-task **offline scaffold** completing with zero infrastructure errors.

The loop contains **no tool bodies**; it is protocol only, so the scaffold runs offline.
Proven to actually iterate (turn count strictly increases; a multi-turn tool response
changes the next action) — the prior rejected iteration was a diagnostic-only object.

---

## 3. M3 — isolated code-execution harness

`henri_sandbox_harness.py` (65,732 B), `tests/contract/test_sandbox_harness.py` (21,795 B).

**Measured on this host:** the live probe reported
`MODE: container-rlimit  isolated: False  surrogate: True`. The **namespace** mode's
constructor probe could not spawn an isolated subprocess, so it correctly refuses
rather than silently downgrading. `/bin/echo` is an external binary inside a namespace;
the probe uses a shell builtin / `sys.executable` instead.

- Typed result: `{status ∈ {PASSED, FAILED, EXECUTION_ERROR, TIMEOUT, VETOED}, returncode,
  stdout, stderr, elapsed_ms, mode, isolated}`.
- A subprocess that cannot start maps to `EXECUTION_ERROR`, **never `FAIL`**.
- 300 s default timeout, configurable; markdown code fences stripped before exec.
- Arithmetic `passed + failed == attempted` enforced.

---

## 4. M4 — long-document ingress

`henri_pdf_ingress.py` (59,318 B), `tests/contract/test_pdf_ingress.py` (31,008 B), 49/49 tests pass.

- Default-OFF behind `HENRI_PDF_INGRESS=1`; byte-identical bypass when absent.
- Backend reported truthfully (`pymupdf`/`fitz` measured; `fitz` import emits a
  deprecation warning and is reported, not hidden).
- Page-level provenance: page number, char offsets, per-chunk sha256 → any answer traces
  to a source page.
- **No raw document text reaches Zone C** — provenance IDs and hashes only. A
  contamination guard fires on the write path, with a **negative control** that
  simulates a text leak and asserts the guard raises.
- Rejects silent projective flattening; fail-closed on missing/corrupt/empty input.

---

## 5. M7 + S0 — evaluation runner and evidence infrastructure

`henri_eval_runner.py` (59,527 B), `henri_eval_infra.py` (12,329 B), 42 + tests.

**S0 (`henri_eval_infra.py`)** exists because of measured defects:

| Component | Defect it prevents |
|---|---|
| `run_output_dir(root, commit, benchmark, run_id)` | two runs sharing one filename **destroyed an earlier verdict**; `exist_ok=False` makes reuse raise |
| `ItemLedger` (flush + fsync per row) | aggregate-only telemetry **lost every row** when aggregation crashed |
| `reconcile(...)` | `passed+failed==attempted`, `attempted+errors==item_count`; a breach is `INVALID_RUN` |
| `sha256_text_lf` vs `sha256_bytes` | a text manifest hashed raw **did not reproduce** from a fresh clone (`core.autocrlf=true`) |
| `scan_contamination` | K-D: benchmark task rows must never reach Zone C or a model store |

**M7 (`henri_eval_runner.py`)** adds the ordered fail-closed pilot gate chain:
static-bundle digest → contamination scan → checkpoint provenance preflight → decoder
fallback source scan → sandbox availability. Each gate returns
`(passed, reason, evidence)`; a failure names the gate and halts. Plus incremental
per-item JSONL, reconciliation, unique run paths, and a `≤16`-item plumbing mode.
**Negative control per gate** (make the gate fail, assert fail-closed).

Nothing under `_archive/invalid_evaluators/` was resurrected or imported.

---

## 6. S1 — SciCode pinned and staged (CPU-feasible progress)

| Field | Value |
|---|---|
| Dataset | `SciCode1/SciCode` |
| Revision (immutable) | `4510f6a6aa27c43fad7b43da2c59602a86e88480` |
| Gated | `False` |
| License | `apache-2.0` |
| API response sha256 | `8f0f3e4063401bd0d5c2a72bd0b20a852b71c51454a2297aa870b83619edc530` |

| File | Bytes | Rows | sha256 |
|---|---|---|---|
| `problems_dev.jsonl` | 279,558 | 15 | `193968aff23b7ed931c8f6d1…` |
| `problems_test.jsonl` | 933,500 | 65 | `38797fef78f434720be6d053…` |

Structure verified (keys only; task text is never printed or persisted to Zone C):
`problem_id`, `problem_name`, `problem_description_main`, `problem_background_main`,
`problem_io`, `required_dependencies`, `sub_steps` (list), `general_solution`,
`general_tests` (list). The 288-subproblem figure corresponds to summing
`sub_steps` across the 65 test problems.

This makes **S1 runnable without funding**: SciCode grades by code execution, which
depends on the M3 harness (built), not on the blocked open-answer egress path.
Gate to advance: scaffold `≤16` subproblems with **zero infrastructure errors**.

---

## 7. M5 and M6 — deliberately NOT resolved

**M5 — world-knowledge boundary.** `REQUIRES_APPROVAL`. AA-Omniscience (6,000 items),
HLE (2,158), and AA-LCR measure **memorized world knowledge**. Zone C holds frozen
**engrammatic priors** — conservation, symmetry, causality/order, locality, information
bounds, containment, quantity, rigidity — as a constraint manifold, explicitly not a
text lake. Closing the 30% General + Scientific-Reasoning region requires either
accepting it as out of scope or authorizing a frozen revision-pinned backbone
(`HENRI_BACKBONE=1`) with matched ablations, a contamination review, and
evaluator-level provenance. **No backbone was enabled. No contract was changed.**

**M6 — funded CUDA.** `BLOCKED`. Vast `50797414` is `exited`, credit `0`. Ports:
`:5434` OPEN (dev Zone C), `:10100` / `:8000` / `:8090` CLOSED. Every remote gate stays
blocked. All work here was designed to be CPU-feasible so that only *execution* waits
on funding.

---

## 8. Honest limits

- **No score. No member is evaluated.** Every AAII v4.3 member remains `NOT_EVALUATED`.
  Nothing in this document is a capability claim.
- A1/A2 are **CPU mechanism probes**, not benchmark results. n = 32 pairs, one seed.
- The M2/M3/M4/M7 harnesses are **plumbing**: they execute and record, they do not
  demonstrate task capability. A green harness is not intelligence.
- The `container-rlimit` mode is an **explicit surrogate** (rlimits + setsid + timeout,
  no network namespace). It is labeled as such in every result.
- The 25%/75% locally-reproducible split is unchanged: only SciCode, Terminal-Bench 4.0,
  and AutomationBench-AA can be graded locally.
- **M5 and M6 are unresolved by design**, not by omission.

---

## 9. Decision requested

1. **A2 tokenizer binding** — decide how token ids become strings for the 32000-way head.
   Options: (a) supply the exact tokenizer the checkpoint was trained against;
   (b) retrain/replace the head against a tokenizer present on this host (llemma 32019 or
   llama-test 32003 are the closest); (c) declare the open-answer region out of scope.
   Without one of these, 40% of index weight stays `BLOCKED_MISSING_TOKENIZER`.
2. **M5 ruling** — accept the 30% General/Sci-Reasoning region as out of scope under the
   engrammatic-priors contract, or authorize the frozen backbone with ablations.
3. **Fund or defer CUDA** — nothing executes remotely until an instance exists.
4. **Promote `carrier/aaii-v43` to `main`?** It is pushed and independently verified
   (`origin/carrier/aaii-v43` = `f71f95bd2ea6a1c44207e3cdc936607f77143f6b`). `origin/main`
   is untouched at `7750abb`.
