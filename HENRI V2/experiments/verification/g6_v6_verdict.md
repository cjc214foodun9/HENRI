# G6 v6 verdict — count-aware codec: multiplicity SOLVED, sequence KILLED (V1)
Date: 2026-09-07. Branch: carrier/g6-count-aware-egress. Sealed @ c523c41.
Prereg: experiments/verification/g6_v6_count_aware_prereg.md (pre-result).

## Measured result (OBSERVED, vast-5090, frozen manifest 45/45 verified, 60 windows)
- 10/10 contract tests PASS; probe RC=0; determinism 60/60.
- V1 exact sequence >= 0.90: **0.0333 -> KILL** (pre-registered kill: V1 < 0.70).
- V2 multiset P >= 0.95 / R >= 0.95: **1.0 / 1.0 -> PASS**.
- V3 OOV abstain: PASS (ABSTAIN_LOW_CONF, no text).
- V4 determinism: PASS (60/60).
- V5 zero fabricated tokens: PASS (0/60 windows).
- V6 multiplicity-exact >= 0.90: **1.0 on 59/60 windows with repeats -> PASS**.

## Mechanism diagnosis (OBSERVED @ c523c41, order_uniqueness + order_diagnosis)
- Q1: true adjacent-bigram evidence strong on 60/60 (order info IS in wave).
- No-repeat windows: unique 1st-order walk (1.0) AND exact beam decode (1.0).
- Repeat windows (59/60): 1st-order graph ambiguous (mean max out-deg 3.83;
  uniq_1st_frac_repeat 0.0678) -> beam picks a wrong valid walk.
- 2nd-order (trigram) constraint: one 36-word repeat window uniquely recovered
  (n_walks_2nd=1, complete) -> positive mechanism signal; BUT a sibling window
  reports n_walks_2nd=0 with cap_hit=0, which contradicts gold-walk validity
  (all true trigrams are written). DISCLOSED DEFECT: the trigram-uniqueness
  diagnostic is not reliable evidence (suspect strong-map/DFS scope bug); the
  2nd-order trigram beam (all-start) scores 0.0/60, consistent with beam
  weakness, not representation absence. v7 must fix the diagnostic before use.

## Conclusion (honest, per evidence classes)
- v6 SOLVES: exact word multiset + exact multiplicity + zero fabrication +
  OOV abstain + determinism over K5-scale frozen real corpus (58298 vocab).
- v6 DOES NOT SOLVE: exact sequence at 30-60 word scale. Root cause =
  decode-side walk ambiguity under repeated words, not wave encoding (order
  information verified present). First-order bigram constraint is insufficient;
  2nd-order shows one positive recovery but its diagnostic is defective.
- Sealed verdict: V1 KILL FIRED. v6 stays default-OFF, carrier = sealed
  negative + mechanism evidence for v7. NO production wiring, NO main push.

## Next (v7 amendment path, requires NEW prereg)
Count-aware high-order walk decode: 3rd-order features + exact walk
enumeration with strong constraints (fix uniqueness diagnostic first),
prereg exact-seq >= 0.90 stratified by repeat/no-repeat, kill V1 < 0.70.
