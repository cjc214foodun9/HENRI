# G7 v7 verdict — high-order walk decode: ACCEPTED (V1 solved, no kill)
Date: 2026-09-08. Branch: carrier/g7-highorder-egress. Sealed @ 0153fc14.
Prereg: experiments/verification/g7_v7_highorder_walk_prereg.md @ 7224b64
(pre-result, sealed 2026-09-07).
Base: origin/main f4dcb69; G6 sealed carrier 8a03f6b (evidence reference).

## Measured result (OBSERVED, vast-5090 @ 0153fc14, manifest 45/45, 60 windows)
Remote wrapper: HEAD=0153fc14a5171964f5fcc8bc30d84197a07df299 (DIRTY=0);
G7_DIAG_RC=0, G7_PROBE_RC=0, G7_VERIFY_RC=0. Contract 10/10 PASS (Linux py3.12).

### SCALE-1 diagnostic (fixed v6 defect)
- gold_valid_frac = 1.0 (60/60). gold_valid_norepeat=1, gold_valid_repeat=59.
- worst_gold_ev = 1.0 (every true adjacent bigram/trigram/4-gram evidence FULL).
- any_cap_hit = 0; uniq_frac_all = 1.0; uniq_frac_goldvalid = 1.0.
=> The v6 diagnostic defect is closed: gold-walk validity is reproduced on
every window, AND every window has exactly ONE valid walk under the 2nd-order
(bigram+trigram) constraint. No 3rd-order escalation needed on 60/60.

### v7 probe (prereg criteria)
| Criterion | Threshold | Measured | Result |
|---|---|---|---|
| V1a exact seq no-repeat | >= 0.95 | 1.0 (n=1) | PASS |
| V1b exact seq repeat | >= 0.90 | 1.0 (n=59) | PASS |
| V2 multiset P / R | >= 0.95 | 1.0 / 1.0 | PASS |
| V3 OOV abstain | 3/3, no text | ABSTAIN_LOW_CONF | PASS |
| V4 determinism | 60/60 | true | PASS |
| V5 zero fabricated tokens | 0 | 0 | PASS |
Kill check: V1b<0.70 F, V1a<0.85 F, any_fabrication F -> NO KILL FIRED.
vocab_size=58298, n_ok=60/60, n_abstain=0, n_cap_hit=0.

## Mechanism conclusion (OBSERVED + DERIVED)
v7 EXACT WALK ENUMERATION over the strong-edge multigraph with 2nd-order
(trigram) constraints resolves every one of the 59/60 repeat windows that v6's
1st-order beam could not (v6 exact seq 0.0333). Key: order info was already in
the wave (v6 Q1 60/60); the missing piece was DECODE-SIDE constraint order
(2nd-order edges + exact DFS instead of beam), plus the fixed gold-walk
validity diagnostic. Multiplicity (V6) and admission (V2) contracts preserved.

## Evidence class (honesty)
REPRESENTATION_CAPACITY_EVIDENCE (self-consistent round-trip over frozen real
corpus). NOT a task score, NOT AAII, NOT production egress. Carrier remains
default-OFF; no DB writes; zero trainable parameters. NO main promotion in this
carrier (separate approval gate per prereg).

## Next
- v7 = ACCEPTED carrier. Promotion decision remains with governance gate.
- G8 carrier (thermo partition calibration, Corberi) is a SEPARATE workstream.
