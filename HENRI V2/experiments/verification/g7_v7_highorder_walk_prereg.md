# G7 v7 pre-registration — high-order walk decode (count-aware, exact enumeration)
Date: 2026-09-07. Carrier: NEXT (proposed; this file seals the amendment pre-result).
Base: origin/main f4dcb69; G6 sealed carrier 8a03f6b (evidence reference).

## Measured boundary that motivates v7 (OBSERVED, sealed g6_v6_kill_receipt.json)
v6 CountAwareCodec: exact seq 0.0333 < 0.70 -> V1 kill; V2/V3/V4/V5/V6 PASS.
Diagnosis @ c523c41: order information IS in the wave (true adjacent bigram
evidence strong on 60/60); no-repeat windows recover exactly (unique walk 1.0,
beam_exact 1.0); 59/60 windows contain repeated words and the 1st-order
strong-edge multigraph is ambiguous (mean max out/in degree 3.83; uniq 0.068).
2nd-order (trigram) constraint recovered 1 repeat window uniquely, but the
uniqueness diagnostic is DEFECTIVE (one window reported 0 walks with cap_hit=0
while gold walk validity is guaranteed) and the all-start trigram beam scored
0.0/60 (beam weakness, not representation absence).

## v7 mechanism (hypothesis to falsify)
Encoding: unigrams + bigrams + trigrams + 4-grams, multiplicity-preserving
(no dedup, no fill, no row-unit normalization) — same spine as v6, higher
order adds next-token context.
Decoding:
  * admission: 16/16 support gate over bounded vocab (v6-verified exact);
  * counts: median per-cell magnitude (v6-verified exact, mult_exact 1.0);
  * sequence: EXACT walk enumeration (no beam) over the strong-edge multigraph
    with constraints of increasing order: edges must pass 2nd-order
    (previous-2 trigram) evidence >= 12/16; if ambiguity remains, 3rd-order
    (previous-3 4-gram) evidence. Deterministic lexicographic tie-break.
  * Pre-requisite gate (diagnostic fix, SCALE-1): the uniqueness diagnostic
    must reproduce gold-walk validity on every window (n_walks >= 1 for all
    60) before any order-k claim. If it cannot, v7 is BLOCKED_DIAGNOSTIC.
  * OK iff exactly one walk consumes every admitted word exactly its count.

## Pre-registered acceptance (same frozen corpus + 60 windows, seed 20260907,
vocab <= 100k, SHA-verified manifest)
V1 exact sequence >= 0.90 stratified: (a) no-repeat windows >= 0.95;
    (b) repeat windows >= 0.90.
V2 multiset P >= 0.95 and R >= 0.95 (v6-established).
V3 OOV abstain 3/3, no text.
V4 determinism 60/60.
V5 zero fabricated tokens (multiset subset).
Kill: V1(b) < 0.70 OR V1(a) < 0.85 OR any fabrication. (Stricter than v6:
capacity ceiling on repeats must be near-exact or the carrier is killed.)

## Evidence class (honesty)
REPRESENTATION_CAPACITY_EVIDENCE (self-consistent round-trip). NOT a task
score, NOT an AAII score, NOT production egress. Separate carrier, default-OFF,
zero trainable parameters, no DB writes, no main push without separate
approval gate.

## Deferred (explicitly out of scope)
AAII v4.2 evaluation requires: hosted endpoint + calibrated egress +
benchmark-staged evaluators + HLE terms acceptance; NOT claimable from this
carrier. ARC task capability requires calibrated semantic action head +
authorized (observation, GameAction, data) trajectories.
