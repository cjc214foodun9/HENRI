# Gate 3.2 Engagement Re-run Verdict (measured NO_ENGAGEMENT again)

**Carrier:** carrier/stage3-coupling @ `1500c10d` (gate32 prereg sealed)
**Run:** vast-5090 GPU-exclusive, clean detached worktree @ `1500c10d`, overlay `7557238908`, `HAS_DSN=True`
**Flags both arms:** `HENRI_ARC_SCORECARD_DELTA=1 HENRI_DELTA_GAIN=1 EXTERNAL_OUTCOME_EFE=1`
**Bounds:** 60 steps, seed 20260908, real env IDs (live Arcade probe)

## Measured (OBSERVED)

| Env | RC | rows | delta_gain_valid | nu_events | progress_events | max levels_completed | Predicate |
|---|---|---|---|---|---|---|---|
| ar25-0c556536 | 0 | 128 | 60 | 60 | **[]** | **0** | NO_ENGAGEMENT |
| lp85-305b61c3 | 0 | 128 | 60 | 60 | **[]** | **0** | NO_ENGAGEMENT |

Mechanism A wiring is consumed (`delta_gain_valid=60/60`, `nu_events=60`),
but **zero authoritative scorecard delta events** occurred in-window on either
"tractable" candidate. Combined with the earlier ka59 result (0 events), the
delta-gain update was never re-weighted by an outcome.

## Verdict: `NO_ENGAGEMENT_PROMOTION_WITHHELD` (stands)

- Mechanism A is NOT falsified; outcome engagement is UNMEASURED.
- User Action-1 precondition (ΔS ≥ 1.0 within window) is NOT met on ar25/lp85/ka59 at 60 steps, seed 20260908.
- Next falsification: env family or window length with in-window `levels_completed` increase; then re-measure.
- Gate 3.4 is **NOT dispatched**: Actions 1+2 have not passed; promotion requires fresh approval; main = `10f5f23` untouched.
- Sealed: `#91d1d697` (this result).
