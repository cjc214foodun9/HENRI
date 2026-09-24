# UHR-04 RUN #7 VERDICT — ka59 env-probe (independent replication)

Instance `52301758`, carrier `dbe7b42`, overlay `799,034,119 B` / `7557238908…`.
Config: `HENRI_SINGLE_ENV=ka59-38d34dbb`, `steps=32`, `tau=0.3500`, one flag differing.
Judged against the frozen run-#7 pre-registration (uhr03_preregistration.md).

## 0. Infrastructure
| item | value |
|---|---|
| BASELINE exit | 0 |
| RFSS exit | 0 |
| egress | `EGRESS_VERIFIED` both, `all_sha_ok=1` |
| stop | `CONFIRMED_STOPPED`, running=0 |
| credit after | $13.8573 (from $14.2041 pre-run) |

## 1. Amendment 1 — the migration WORKED (this is the headline)
`ka59-38d34dbb` is REACTIVE: `frame_changed = 32/71` in BOTH arms, versus
`ft09`'s measured `0/16`. The retired environment was static; the migration target
moves. This is the first live pair in the program whose frame is not a cursor band.

Per-action changed-cell counts on the migrated env are action-contingent
(historical corpus: means 15.73 / 17.34 / 18.60 / 11.17, six distinct values).

## 2. Preconditions (G1-G5) — ALL PASS
| gate | requirement | measured | verdict |
|---|---|---|---|
| G4 | both arms produce records | 71 / 71 | PASS |
| G5 | flag-OFF emits NO extero keys | BASELINE extero = 0 | PASS |
| G1 | the frame actually moves | `frame_changed = 32/71` | PASS |
| G2 | guard executed | `guard_state.updated = true`, 32/32 | PASS |
| G3 | pooled channel reached | `extero_info.status = OK`, 32/32 | PASS |

G5 is the default-path identity control: with `HENRI_UHR02_EXTERO_GATE=0` the
pooled channel emits nothing, so the flag-OFF path is unchanged.

## 3. FORM B on the live path
```text
pooled_status              OK 15 | UNAVAILABLE_SUPPORT_TOO_SMALL 17
pooled_n_channels          {1: 17, 2: 9, 19: 6}
pooled_own_is_min          True 15 | (unscored) 17
pooled_margin_above_band   True 15 | (unscored) 17
pooled_margin              n=15  min 0.0267257  max 0.0475249   (ALL POSITIVE)
C1 own = argmin            15/15   (recomputed from raw delta_pooled_all)
C2 margin > band           15/15
C3 own varies              15 distinct values -> LEAVES {0.0}
spread_above_band          {True: 1, False: 3, rest unscored}   <- CONTENT verdict
magnitude_only_risk        {True: 31, False: 1}
role_coherence             0.004608 .. 0.037214
|Tr(Uc^dag Ut)|            0.645598 .. 3
```

**VERDICT: PASS — ACTION IDENTIFICATION, not content discrimination.**
On the 15 records where the pooled domain was computable, the true action's
predicted transition was the argmin of its competitor set in **15/15**, with a
margin above the noise band in **15/15**. That is the pre-registered anticipated
outcome #2 in uhr03_preregistration.md, exactly as written before this data existed.

**Scope limit, stated not buried.** Only **1** of 32 records showed a
content-bearing competitor spread (`spread_above_band = True`). The remaining
competitors compose to the identity at the observed cells, so the live evidence
establishes *which* action was taken, **not** that the system discriminates *what*
the action did. Content discrimination still rests on the LOCAL hard control
(`own 0.03769 < S_hard(same cells,+2) 0.509042 < untrained 0.532079`).
`magnitude_only_risk = True` in 31/32 is consistent with that: the residual is
dominated by the trace because the competitor set is identity-only.

**A newly measured limitation (not a defect):** 17 of 32 records returned
`UNAVAILABLE_SUPPORT_TOO_SMALL`, i.e. only ONE channel moved, so no competitor set
exists and the record is UNSCORED. The pooled domain needs >= 2 support channels.
That caps the scored fraction at ~47% on this env. It is a DATA property of ka59's
small step-to-step change, not a threshold that can be tuned away.

## 4. Kill budget
**NOT consumed.** FALSIFIED requires two consecutive runs that pass G1-G6 with a
CONTENT-BEARING competitor set **and** fail both C1 and C2. No such run has
occurred; this run passed both.

## 5. Amendment 10 admissibility
Counts as an **independent replication**: the environment differs from every prior
UHR run (`ka59-38d34dbb` vs `ft09-0d8bbf25`). Run #6 did not qualify (identical
env and action sequence); this one does.

## 6. Reproduce
```bash
# evidence (local egress, no instance needed)
%LOCALAPPDATA%/Temp/uhr03_egress/telemetry_uhr03_BASELINE/production_run_1790233058.jsonl
%LOCALAPPDATA%/Temp/uhr03_egress/telemetry_uhr03_RFSS/production_run_1790233764.jsonl
python %LOCALAPPDATA%/Temp/uhr04_reduce7c.py     # C1/C2 from raw delta_pooled_all
python %LOCALAPPDATA%/Temp/uhr04_ledger_control.py
```
