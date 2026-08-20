# Phase 8.39 — Reward-shaped ranking verdict (test-time learned positive-exemplar prior)

Status: **FALSIFIED (pre-registered kill fired: solved < 3/50)**
Branch: `phase839/humaneval-wave-ast` @ `0066e9a`
Date: 2026-08-20

## Result (OBSERVED, CUDA RTX 5090, n=50, d=65,536)

| Arm | solved | expressible | infra_errors | items_reordered |
|---|---|---|---|---|
| Baseline (no flag, `79222a7`/`171c21c`) | **2/50** | 50 | 0 | 0 |
| `--reward-rank` (`0066e9a`) | **1/50** | 50 | 0 | **25** |

PASS with reward-rank: HumanEval/23 (`return len(string)`). Baseline passes /23 + /35 (`return max(l)`) were NOT reproduced; the /35 correct candidate was de-ranked by the /23 exemplar direction and fell below the 12-attempt cutoff.

## Mechanism (DERIVED)

- `items_reordered = 25` proves the re-ranking actively engaged — the lever is not inert.
- Cross-item transfer FAILS: exemplar directions from one item's verified solution do not generalize to another item's transformation-relative space at current codec fidelity (distinct-option cosine ~0.35–0.55 geometry class).
- The earlier audit (inert ranking: `decode(prompt_wave, prompt_wave)` → zero target direction) is the background truth: there is no per-item predicted wave in the grammar-enumeration path, so no positive per-item ranking signal exists; the exemplar prior was the only candidate and it is item-specific noise.

## Pre-registered gate

- Kill: solved >= 3/50 → PASS, promote flag. OBSERVED 1/50 → **KILL FIRED**. Flag remains default-OFF.

## Disposition

- `--reward-rank` stays default-OFF, unpromoted. Honest negative promoted to the campaign registry.
- Next lever (representation change, not codec parameters): **trained-decoder ranking head** — use the trained 32k unbinder checkpoint (exists, 799 MB, `henri_decoder_checkpoint.pt`, SHA `75572389…`) as a candidate SCORER (wave → logits → per-token log-likelihood of candidate code), replacing inert geometry ranking with a real learned signal. Token-decode GENERATION remains falsified; scoring is a distinct mechanism.

## Evidence artifacts

- Scorecard: `experiments/verification/humaneval_rewardrank_839_scorecard.json` (item-level).
- Log: `/root/telemetry_logs/humaneval_reward_rank_839.log`.
- Governance event: appended (see event store).
