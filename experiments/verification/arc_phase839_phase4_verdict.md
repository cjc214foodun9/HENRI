# Phase 4 Verdict — Egress Hardening & REPL Feedback Audit

Spec: roadmap Phase 4 (M5: AST-node logit masking → 0 out-of-vocab errors; REPL feedback → EFE)
Status: **MASK PASS / REPL-EFE NOT IMPLEMENTED** — grammar-mask fail-closed hardening PASS (one defect fixed, contract tests 3/3); REPL-feedback → EFE planner causal path NOT IMPLEMENTED (`--reward-rank` is a prior-falsified heuristic, not an implemented EFE planner).

## 1. Audit-first findings (OBSERVED, live code @ 448926c)

| Claim (roadmap) | Live code | Verdict |
|---|---|---|
| "Implement AST-node masking on the egress head" | `HENRIASTGrammarMask` ALREADY EXISTS (94 l.); wired in `henri_decoder.py` (autoregressive legacy path, step 0–2 rules + Rules 4/5) and `henri_fused_triton_cuda_graph_runner.py` (GPU bitmask path) | EXISTS — no re-implementation needed |
| Mask on the HumanEval score path? | `humaneval_wave_ast_runner.py` has ZERO `HENRIASTGrammarMask` references — score path = grammar enumerator (`WaveASTDecoder._instantiate`) + wave ranking + `SecurePythonSandbox` | NOT on score path (by design) |
| "Connect egress to seccomp-blocked sandbox (rlimit_as=512MB, rlimit_cpu=2s)" | `SecurePythonSandbox` mode="container-rlimit": `RLIMIT_AS = memory_bytes` (1 GiB default, parameterized) + `RLIMIT_CPU = timeout_sec` (+1) | EXISTS — bounds parameterized; values differ from roadmap (1 GiB vs 512 MB) |
| "Feed REPL execution feedback into active inference EFE" | `--reward-rank` exemplar seeding from sandbox-verified favorable outcomes (default-OFF); no EFE/planner consumer on the HumanEval score path | **NOT IMPLEMENTED** — `--reward-rank` is a prior-falsified heuristic (1/50, Gate B sealed `3654b60`), NOT a causal REPL→EFE planner path |

## 2. Defect found + fixed (all-masked fail-closed)

**Defect:** `mask_logits_for_step` returns a fully-masked logits vector (argmax over all −1e9) when the vocabulary lacks the mandatory token for the step — silently emits garbage token id 0.

**Fix (`448926c`):** `GrammarMaskAllMaskedError(RuntimeError)` raised at EVERY mask return site via `_assert_unmasked()`. Callers treat it as a typed execution error — never a silent emission.

**Contract tests (`tests/unit/test_ast_grammar_mask_all_masked.py`, 3 passed):**
1. step-0 all-masked vocab → raises `GrammarMaskAllMaskedError`
2. normal vocab step 0 → single unmasked token, argmax == 0
3. 1D and 2D ([V] and [B,V]) logit shapes preserved

Local suite: 173 passed / 1 skipped (was 170/1; +3 new).

## 3. Pre-registered factorial — NOT executed as benchmark arms

Reference 3 / skill discipline requires a factorial (control, mask-only, REPL-only, both) before promotion. The audit shows:
- The mask is NOT on the HumanEval score path → a mask-only arm on the score path would be a no-op by construction (vacuous).
- The REPL-feedback arm (`--reward-rank`) is already FALSIFIED at Gate B with a sealed verdict + governance event.

**Decision: no new benchmark factorial** — the two roadmap Phase 4 levers are either off-path by design (mask) or already falsified (REPL feedback). Adding arms would be mock-loop theater. Recorded as `BLOCKED_BY_PRIOR_FALSIFICATION` for the feedback arm and `NOT_ON_SCORE_PATH` for the mask arm.

## 4. Evidence classes

- `OBSERVED`: audit greps (mask sites, runner references, sandbox rlimits), contract test results
- `FALSIFIED` (prior, standing): `--reward-rank` 1/50 Gate B (event `a7b93863…`/`3654b60` registry)
- `BLOCKED / NOT_IMPLEMENTED`: REPL-feedback → EFE planner causal path (no live consumer on the score path)
- `HYPOTHESIS`: roadmap's claim that masking eliminates runtime crashes on the score path (mask not consumed there)
