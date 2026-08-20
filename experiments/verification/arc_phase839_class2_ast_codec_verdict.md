# Phase 8.39 — Class 2.0 AST-Structured Codec + MBPP Codebook verdict

Status: **FALSIFIED at pre-registered proxy Gate A (oracle ranks > 5)**
Branch: `phase839/humaneval-wave-ast` @ `b00255d` (base for this experiment)
Date: 2026-08-20

## Premise correction (OBSERVED, controlling)

The user's brief stated Levers 2.1 (`--codec-ast-bound`) and 2.2
(`--codebook-mbpp-seed`) as "implemented and verified". Audit against live
code across ALL local branches, origin refs, and the remote worktree:
**0 grep hits for either flag; no `ASTqFHRREncoder` / `qfhrr_ast_kernel`
symbol exists in any ref.** The four inbox documents are the *implementation
spec*, not an implementation. Launching Gate A against the old char-ring
codec would have measured the falsified baseline (mock loop). The levers were
implemented faithfully from the docs, then gated.

## Phase 1 + 2 (implemented, committed as sealed experiment artifacts)

- `HENRI V2/qfhrr_ast_kernel.py` — `ASTqFHRREncoder`: deterministic SHA-256
  node-type phase vectors; fractional position binding (depth × P_depth +
  child × P_child) mod 256; complex phasor accumulation; Z_256^D uint8 out.
- `HENRI V2/scripts/staging/ingest_mbpp_codebook.py` — canonical MBPP
  ingestion (sha `ccf64ceae9c5403b`), provenance SHA-256 per record, HumanEval
  contamination guard, file-backed `zone_c_mbpp_codebook.pt` payload
  (deviation from the plan's DB table: file-backed avoids production DDL
  without approval; recorded in the module docstring).

Kernel mechanics probe (OBSERVED, CPU d=2048): deterministic; round-trip
phase-cos = 1.0000; node-type vector orthogonality cos(Return, BinOp) =
0.0086. **Carrier dominance measured**: raw phase-cos between ANY two
unrelated programs = 0.59–0.64 (shared FunctionDef/Return/arg skeleton
dominates the superposition). Decoder-style linear v_rel ranking in AST-wave
space is degenerate (sims ≈ −1.0, std 0.0001) — not the spec mechanism.

## Proxy Gate A oracle (OBSERVED, CPU d=2048, real candidate pool, real codebook)

Mechanism per spec (Lever 2.2): rank each grammar candidate by mean raw
phase-cosine vs the MBPP codebook attractor bank (N=100 waves from canonical
`mbpp.jsonl`, sha `ccf64ceae9c5403b`). Candidate pool = the runner's full
grammar set (71 candidates/item, same `WaveASTDecoder._instantiate`).

| Item | correct body | rank | window (≤5) |
|---|---|---|---|
| HumanEval/23 | `return len(string)` | **39/71** | FAIL |
| HumanEval/35 | `return max(l)` | **29/71** | FAIL |

- sim std 0.026 ⇒ mechanism engaged (non-inert run; not a false negative).
- top-3 for both items: `return t`, `return c` — generic loop-sum/loop-count
  bodies frequent in MBPP; the codebook mean-cosine selects shared skeleton
  shape, not program semantics.

## Pre-registered gates (plan doc `HENRI-PLAN-CLASS2-TRANSITION-2026`)

- Gate A: both passing solutions rank ≤ 5 → **FAILED (39, 29)**. Kill fires.
- Gate B (50-item sweep): **SKIPPED per protocol** — GPU conserved.
- CUDA D=65,536 Gate A: **NOT RUN** — proxy killed it. Direction argument
  (DERIVED, not OBSERVED): node-type vectors are already near-orthogonal at
  d=2048; higher D increases random-term averaging, making the shared-skeleton
  carrier MORE dominant in relative terms. No credible path to ranks ≤ 5 at
  D=65,536; the failure mode is structural (superposition carrier mass), not
  dimensional.

## Disposition

- Components stay committed as the sealed experiment artifact, default-OFF.
  **No runner wiring** (`--codec-ast-bound` / `--codebook-mbpp-seed` NOT
  added): a killed path does not become a dead flag in the production runner.
- Lever class: the "representation change" class is now FALSIFIED for this
  construction at the proxy gate. Standing scores unchanged (HumanEval 2/50,
  GPQA 0.298, MMLU 0.2598 — all OBSERVED).
- Governance event appended; registry updated.

## Evidence

- Probe runs: this session (terminal, CPU, d=2048; logs quoted above).
- Files: `HENRI V2/qfhrr_ast_kernel.py`, `HENRI V2/scripts/staging/ingest_mbpp_codebook.py`.
- Data: `mbpp.jsonl` sha `ccf64ceae9c5403b`; HumanEval canonical (runner cache).
