# UHR-03 — local discrimination evidence (zero GPU)

All numbers here are `OBSERVED` from local CPU runs. They exist to decide whether a
live paired A/B is worth the GPU spend, and to say exactly what a live pass would and
would not prove.

## 1. The shipped helper, hard comparator restored

`G.pooled_domain_statistic` — the function the live runner calls — on a fixture where
the invalid comparator shares the true action's support and differs only in content:

| anchor | cells | transition |
|---|---|---|
| true action (2) | rows 4-5 | `+1` |
| HARD negative (3) | rows 4-5 | `+2` |
| EASY anchor (4) | rows 10-11 | `+1` (disjoint) |

7 seeds, `K=8192`, `band = 2.470529e-03`:

| arm | PASS | margin mean | margin min | ties |
|---|---|---|---|---|
| STABLE | 7/7 | +0.47302 | +0.47211 | 0 |
| VARIED | 7/7 | +0.12096 | +0.12041 | 0 |

Small margins diverge from the reimplementation (v3 gave min +0.12138 on VARIED/PRE)
because the shipped helper reads theta after training while v3's PRE arm snapshots it
before folding the observation in. Same conclusion, different convention.

**Comparator integrity:** HARD scores `0.50952` and EASY scores `0.53037`. The HARD
comparator sits BELOW the do-nothing baseline because it shares support and therefore
partially matches. If HARD had equalled EASY, the comparator-swap defect of commit
`e32ff99` would have returned; it did not.

## 2. Why an earlier probe showed 2/5 — the stimulus, not the comparator

Commit `1ef5892` recorded "true action wins 2 of 5 seeds". That fixture cycled the
absolute colour offsets between training steps (`1+2k` vs `2+2k`), and
`encode_su3_color_field` is **not a homomorphism in colour**: shifting a cell by `+1`
produces different relative displacements at different absolute colours (measured
`||disp_a - disp_b||max = 1.448e+00` for two `+1` shifts at different absolute
colours). A cycling stimulus therefore makes each training displacement a *different*
group element, the EMA averages them, and the store stops representing any single
transition. With the absolute colours held fixed, the displacement is one fixed
element and the store converges.

This is why the `T` magnitude — not the saturated win rate — carries the information:

| stimulus | `T` (true action) |
|---|---|
| STABLE | 0.0374 |
| VARIED | 0.4094 |

An ~11x effect. `wins = 7/7` is uninformative once every arm saturates; `T` is not.

## 3. What a live A/B can and cannot prove

The live store holds one theta per action, learned at the cells that action changed.
At the observed channels, the true action has support and the others largely do not,
so the live `invalid_min` is dominated by *support overlap* (the do-nothing baeeline).
A live pass therefore shows the mechanism can identify WHICH action was taken; it does
NOT isolate content discrimination, because no live comparator shares the true action's
support with different content.

The local hard-comparator test in section 1 is therefore **strictly stronger** than the
live test. A live run is still worth one pass — it exercises the pooled plumbing in
situ and supplies a live margin for the record — but a live pass must not be reported as
evidence of content discrimination.

## 4. Standing caveats

- `STABLE` is partly in-sample: the store trains on the transition it is scored against.
  `VARIED` is the conservative proxy, and it still passes 7/7.
- The live `stalled` / `stopped_early` triggers are NOT wired; contract tests pass but
  the runtime path is unexercised. Treat them as unverified.
