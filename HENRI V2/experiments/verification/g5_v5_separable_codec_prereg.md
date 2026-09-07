# G5 v5 pre-registration — SeparableCodec (full-entropy cells) + exact decode
Date: 2026-09-07. Branch: carrier/g5-egress-wavepacket. Base: origin/main 88fde19.

## Measured falsification that motivates v5 (sealed 2026-09-07)
K5 CompositionalTextCodec cell map depends only on x mod 8192:
  probe distinct_cells_reachable = 8192 of 65536;
  tokens '013' and 'duryee' share h mod 8192 -> IDENTICAL 16-cell signatures
  (global probe: cells_equal=True, overlap=16).
At K5 vocab scale (58,298 words), support gate accepted 1,554 words for one
40-word window (30 true + 1,524 false) -> deterministic inversion of K5 waves
FALSIFIED (span P=0.0052, R=0.012, OOV guard failed). K5 is non-invertible.

## v5 mechanism
SeparableCodec derives each of 16 cells from the FULL feature hash via
splitmix64(h ^ (s+1)*GOLDEN64) % 65536. Effective address space = 65536;
two features collide on all 16 cells only on a full 64-bit hash collision
(negligible). Encode keeps the [8192,8] row-unit payload + 2000-d proj
(K5-shaped, ARM-R parity). Decode = exact support-membership gate
(all 16 cells present AND sign-matched) over a bounded vocab, then bigram
beam-DP for order; emit OK or ABSTAIN. Zero trainable parameters.

## Scope note (honesty)
v5 decodes ONLY waves it encoded. It cannot decode existing K5 corpus waves
(wave_payload rows were produced by the collapsed-map codec). Corpus benefit
requires a future staged re-ingest with the v5 codec; this carrier ships the
codec + tests + exact-decode evidence only.

## Pre-registered acceptance (60 windows, 6 sources, 30-60 words, K5 vocab)
V1 exact-sequence recovery >= 0.90.
V2 token P >= 0.95 and R >= 0.95 on recovered windows.
V3 OOV guard: 3/3 abstain (no fabricated text).
V4 determinism: 3/3 identical.
V5 false-hit rate: recovered word sets contain 0 false positives.
Kill: V1 < 0.70 OR any fabricated token in abstain paths.

## Resource limits
CPU/numpy (no CUDA needed for the codec); remote probe on Vast canonical env;
read-only; no DB writes; no main push before approval gate in this carrier.

## Seal
Prereg written BEFORE v5 results. Commit order: prereg + codec + tests, then
probe, then receipt.
