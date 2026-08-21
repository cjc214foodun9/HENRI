# Path B Verdict: PATH_B_GATE_A_FALSIFIED — 2026-08-21

## Status
SEALED. Path B (supervised semantic codec, `981da1e..e25531e`) is FALSIFIED at
the pre-registered Gate A. Gate B was SKIPPED per protocol (GPU compute
conserved). Implementation commits reverted; verdict + evidence preserved.

## Pre-registered gate (class4_path_b_design.md, spec packet)
Gate A on RTX 5090, D=65,536, checkpoint `models/path_b_codec.pt`:
- Condition 1: HumanEval/23 AND /35 oracle rank <= 5 / 71.
- Condition 2: cosine separation (true vs best-other) >= 0.25.
- Kill: failure at either condition -> seal `PATH_B_GATE_A_FALSIFIED`,
  revert, halt (no Gate B).

## Observed (Gate A probe, ckpt sha 40d82326dde1e3ff, dataset b796127e635a67f9)

| Target | oracle_rank | pool | true_cos | best_other_cos | margin | c1 | c2 |
|---|---|---|---|---|---|---|---|
| HumanEval/23 | 31 | 71 | 0.9105 | 0.9944 | -0.0838 | FAIL | FAIL |
| HumanEval/35 | 32 | 71 | 0.8972 | 0.9921 | -0.0949 | FAIL | FAIL |

## Mathematics of the failure (DERIVED)

1. **The semantic signal exists but is NOT discriminative.** true_cosine is
   high (0.91/0.90) — the learned codec DOES recognize the true solution near
   the encoded goal. But best_other_cosine is even higher (0.994/0.992), so
   multiple lookalike grammar candidates outscore the oracle. The codec's
   ranking metric collapses: nearest-other beats true by ~0.08-0.09.
2. **Chance-rank reference:** uniform random rank expectation in a 71 pool is
   36; observed 31/32 is marginally better than chance but far below the
   <=5 requirement — the MBPP-trained metric does NOT transfer to HumanEval
   discrimination.
3. **Mechanism hypothesis (HYPOTHESIS, consistent with corpus consult):**
   mean-pooled lexical + AST-type embeddings are dominated by shared surface
   token mass between the long prompt/docstring and lookalike candidates; the
   short true body contributes little. The contrastive positives (rename
   variants only) taught rename-invariance, NOT lookalike discrimination —
   hard negatives from the grammar pool were absent. This is the same
   carrier-dominance class as Gate A' (8.39 kill), now measured inside a
   learned codec instead of a random ring.
4. **B1 held-out contrastive val acc = 0.7708** (chance 0.5) — non-collapse on
   MBPP renames, but insufficient for cross-dataset ranking transfer.

## Corpus consult record (INFERRED — not telemetry)
The primary bank (ca4bb787, 217 sources) supports differentiable HRR-VQ
codebooks with a learned codebook + complex unit-magnitude projection (pi),
and notes FHRR binding is commutative (symmetric-binding loss for hierarchy).
Forward design must train the discriminator on GRAMMAR-POOL lookalikes
(hard negatives), not rename variants only, and apply an IDF-style token
weighting to defeat surface-carrier dominance.

## Reverted commits
- e25531e fix(accuracy): Path B path consistency
- 0cc1d3e fix(accuracy): Path B device-safe RNG
- 981da1e feat(accuracy): Path B supervised semantic codec (Class 4.3)

Local suite returns to 602 passed / 3 skipped. Remote main untouched
(69b338d). Evidence: Gate A probe JSON (this doc), checkpoint sha
40d82326dde1e3ff (4,811,856 bytes, overlay artifact), remote suite log
7373d5563d1a929d (608 passed / 4 skipped at e25531e — software integrity
does NOT override the failed external gate).

## Next (requires new pre-registration; NOT auto-authorized)
Path B2: same codec architecture + hard-negative grammar-pool contrastive
training + IDF token weighting; Gate A re-run identical conditions. This is
a materially new semantic representation change (new training signal), not a
ranking-lever reopen.
