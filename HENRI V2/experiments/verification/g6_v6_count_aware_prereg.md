# G6 v6 pre-registration — count-aware codec + directed bigram chain decode
Date: 2026-09-07. Branch: carrier/g6-count-aware-egress. Base: origin/main f4dcb69 (G5 tip).

## Measured failure that motivates v6 (sealed g5_v5_kill_receipt.json, 2026-09-07)
v5 SeparableCodec: exact sequence recovery 0.025 < 0.70 -> pre-registered V1 kill.
Root causes (direct source reads, v5/k5 codec):
  (1) features_of() dedups all n-gram features -> multiplicity destroyed before
      wave accumulation;
  (2) per-row L2 normalization erases cross-row magnitudes (count channel lost);
  (3) v5 decode beam forbids repeats (`if w in path: continue`) -> repeated words
      unrecoverable by construction.

## v6 mechanism (NEW codec, not a patch)
Encoding:
  * feature lists preserve multiplicity: all word unigrams in order + all
    adjacent word bigrams in order (NO dedup);
  * wave accumulation per occurrence: acc[cell] += sgn(cell) per occurrence,
    so |value at cell| = occurrence count of the feature's cell set
    (collision noise bounded; median over 16 cells is the count estimate);
  * payload is [8192,8] float32, NOT row-unit (documented contract deviation:
    no fill, no per-row normalization; zero rows allowed) — retrieval proj
    unchanged (2000-d, features_of ngram 2, L2).
Decoding:
  * admission gate: all 16 cells sign-matched (exact support; a cell value can
    never flip sign because every write adds sgn(cell) — the gate is
    near-vacuous for present words and near-impossible for absent words);
  * count estimate per word: median of per-cell est values, rounded, clamped [1,64];
  * sequence: directed bigram chain — edge evidence = matched/16 for "b:a b"
    cells; supported iff >= 12/16; beam DP (width 8) over sequences using each
    word EXACTLY count[w] times, maximizing total supported-edge evidence;
    deterministic tie-break (lexicographic);
  * OK iff the best path consumes every admitted word exactly its count;
    otherwise ABSTAIN_NO_ORDER (never partial text; never fabricated).
Zero trainable parameters. Deterministic. Fail-closed.

## Evidence class (honesty)
The probe is a SELF-CONSISTENT encode->decode round-trip: it measures
REPRESENTATION_CAPACITY_EVIDENCE (wave -> text invertibility of THIS codec),
NOT a task score, NOT an AAII benchmark score, NOT a production egress claim.
No DB writes, no production wiring, no main push inside this carrier.

## Frozen corpus (pinned re-ingest discipline)
experiments/verification/g6_k5_freeze_manifest.json pins SHA-256 of every
/workspace/k5-sources file (captured 2026-09-07, vast-5090). The probe
RE-VERIFIES each source SHA at runtime; mismatch -> BLOCKED_MUTATED (no
evidence, exit 2). This replaces the post-ingest-mutated corpus ground truth
(governance finding 2026-09-07) with a pinned, verifiable snapshot.

## Pre-registered acceptance (60 windows: 6 sources x 10, 30-60 words, seed 20260907, K5 frozen corpus, vocab <= 100k)
V1 exact sequence recovery >= 0.90 on returned-OK windows.
V2 multiset token precision >= 0.95 and recall >= 0.95 (counts matter).
V3 OOV guard: 3/3 abstain (no text).
V4 determinism: 60/60 identical decode.
V5 fabricated tokens = 0 (output token multiset subset of ground truth multiset).
V6 multiplicity-exact >= 0.90 on windows containing repeated words (pred counts == gt counts).
Kill: V1 < 0.70 OR any fabricated token.

## Resource limits
CPU/numpy only (codec is vectorized numpy; no CUDA required); read-only probe
on the canonical Vast env; no DB writes; no main push before a separate
approval gate.

## Seal
Prereg written BEFORE v6 implementation results. Commit order: prereg +
freeze manifest + codec + tests + probe -> remote probe -> receipt.
