# UHR-03 VERDICT — kill-run #5: FORM B SEPARATED on the live path

Instance `52301758` (RTX PRO 5000 Blackwell), carrier `eb988de`, overlay
`799,034,119 B` / sha16 `75572389083455a3`. Env pinned `ft09-0d8bbf25`.
`SCALE num_blocks=8192, d_model=65536`. Both arms `exit=0`. `EGRESS_VERIFIED`
for both arms, `CONFIRMED_STOPPED`, credit `14.8157`, 0 instances running.

This is the **first** run in which every precondition passed and both arms
completed with the pooled FORM B path exercised.

## 1. Preconditions — all PASS

| gate | requirement | measured | result |
|---|---|---|---|
| G1 store populated | `target_theta_norm > 0`, not stalled | `3.245417356491089`, `stalled=False` | PASS |
| G1b frame moved | `> 0` probes changed | `16/16` | PASS |
| G2 guard executed | `updated is True` | `16/16` | PASS |
| G3 FORM B defined | `status == "OK"`, `n_channels >= 2` | `OK`, `n_channels = 4` | PASS |
| G4 both arms exit | `exit == 0` | `0 / 0` | PASS |
| G5 flag-OFF identity | arm A emits no pooled key | BASELINE `pooled_records = 0` | PASS |
| G6 non-trivial ref | not identity-only | see section 3 | PARTIAL |

## 2. Criteria — C1, C2, C2', C3 all PASS

`band = 2.470529e-03` (`sampling_band(8192)`).

| criterion | requirement | measured | result |
|---|---|---|---|
| C1 own is argmin | `own_is_min` in `>= 50%` of scored records | **16/16 (100%)** | PASS |
| C2 margin above band | `margin_above_band` in `>= 50%` | **16/16 (100%)** | PASS |
| C2' independent | recomputed from raw `own < invalid_min` | **16/16** | PASS |
| C3 owns vary | `> 1` distinct value | `distinct = 16` | PASS |

Measured values (`OBSERVED`):

```
delta_pooled_own  : distinct=16  min=0.449260  max=0.470383
delta_pooled_inv  : distinct=16  min=0.491545  max=0.511880
margin            : mean=+0.042074  min=+0.041497  max=+0.042304   (16.8x .. 17.1x band)
```

## 3. SCOPE LIMIT — state it, do not overclaim

Within each record, **all seven invalid actions share one value** (first record:
`{0: 0.51188, 1: 0.51188, 2: 0.470383, 3: 0.51188, ...}`). That shared value is the
do-nothing baseline: those actions changed *different grid cells*, so at the observed
channels they compose to the identity. The live competitor set is therefore
**identity-only**, and this run demonstrates **action IDENTIFICATION on the live
path** — the mechanism can tell WHICH action was taken — not content discrimination.

Content discrimination is isolated by the local hard-comparator test, where the
competitor shares the true action's support and differs only in content:

```
pooled {'0': 0.532079, '1': 0.532079, '2': 0.03769, '3': 0.509042, '4': 0.532079, '5': 0.532079}
own=0.03769 < S_hard(same cells, +2)=0.509042 < untrained=0.532079   margin=+0.471352
```

That ordering (`own < S_hard < untrained`) is the content-bearing result. It is
`OBSERVED` locally; the live run supplies identification, not this.

Channel semantics are now confirmed: `n_channels = 4` at `[4092, 4093, 4094, 4095]`,
i.e. channel index == row-major grid cell, so only four cells moved in that step.

## 4. Reducer defect and its correction (mine)

`uhr03_reduce_pooled.py` v1 judged this run as **"C1 0/16, kill strike 1 of 2" — a
FALSE NEGATIVE.** Two causes, both mine:

1. **Key-name mismatch.** `pooled_domain_statistic` returns UNPREFIXED keys
   (`own_is_min`, `margin`, `margin_above_band`). The runner emits a PREFIXED copy
   alongside (`pooled_own_is_min`, `pooled_margin`, ...). v1 selected the dict that
   carries `n_channels` — the helper's return, unprefixed — and then read the
   prefixed names, which live only on the parent. Every read returned `None`.
2. **Double count.** `find_pooled` matched parent AND child, so 16 records yielded
   32 dicts and the denominator was wrong.

Fix: R1 record only the helper's return dict (identified by `n_channels`); R2 accept
both names and report which supplied each field. Re-run on the same payload:

```
field names resolved from: ['band','delta_pooled_invalid_min','delta_pooled_own',
                            'invalid_minus_own','margin','n_channels']
C1 own is argmin  : 16/16 (100%)      C2 margin > band : 16/16 (100%)
C2' own < inv_min : 16/16             C3 owns vary     : True
```

The inline extractor also raised `TypeError: '<' not supported between int and
NoneType` twice (run #3 and run #5) on the same optional-key sort. A reducer that
misreads its own payload is the same defect class as a gate that cannot fail.

## 5. Comparator discipline (carried forward)

An earlier local probe reported `wins = 7/7` against a comparator that had **no
support** at the observed channels, so `S == B` to six decimals in every row. A
comparator that cannot win is not a comparator. Two numbers being bitwise equal in a
noisy system is a definition leaking through, not a finding. v3 and the shipped-helper
probe restore the HARD comparator (same support, different content), and that is the
control the numbers above rest on.

## 6. Cost

| item | value |
|---|---|
| instance | `52301758` stopped by the chained driver, `CONFIRMED_STOPPED` |
| credit | `14.8157` |
| running | `0` |
| storage | 180 GB across 3 stopped disks = `$1.3320/day` |

## 7. Status and next falsification

**Kill budget: 0 of 2.** FORM B did not fail to separate; it separated. One live pass
is `N = 1`, so this is a **pass, not yet a reproduced pass**. The pre-registered kill
requires two *consecutive failures*; symmetrically, promotion should require a
reproduced pass.

Next, cheapest first:

1. **Reproducibility run (#6)** — same carrier, same pin, same 32 steps, flag-OFF vs
   flag-ON. Confirms the live margin is stable rather than a single draw. Cost ~`$0.11`.
2. **Content-isolating live comparator** — an action with the SAME support but
   different content. This cannot be forced; it requires the environment to produce
   such a transition. Until then the live path evidences identification only.
3. `origin/main` stays at `adcc24e`. Nothing promoted.

---

## ADDENDUM — kill-run #6 (robustness check) and the scope of the pass

Carrier `7a3f74f`, instance `52301758`, env `ft09-0d8bbf25`, both arms `exit=0`,
`EGRESS_VERIFIED` both, `CONFIRMED_STOPPED`, credit `14.6292`.

### Result (OBSERVED)

| gate / criterion | run #5 | run #6 |
|---|---|---|
| G1 `target_theta_norm` | `3.245417356491089` | identical |
| G1b frame moved | 16/16 | 16/16 |
| G2 guard `updated` | 16/16 | 16/16 |
| G3 `status` / `n_channels` | `OK` / 4 | `OK` / 4 |
| G4 both arms exit | 0 / 0 | 0 / 0 |
| G5 flag-OFF identity | BASELINE 0 pooled keys | BASELINE 0 pooled keys |
| C1 `own_is_min` | 16/16 (100%) | 16/16 (100%) |
| C2 `margin_above_band` | 16/16 (100%) | 16/16 (100%) |
| C2' `own < invalid_min` | 16/16 | 16/16 |
| C3 distinct `own` | 16 | 16 |
| margin mean | +0.042074 | +0.042085 |

By Amendment 9's table this is `PASS REPRODUCED`. By measurement it is a **numerical
robustness check**: identical env, identical action sequence, identical channel sets,
identical `target_theta_norm`, no seed, 1/16 records bitwise identical. See Amendment 10.

### What the pass establishes

The live path identifies the taken action: `own_is_min = True` in 16/16 records in two
runs, with `margin ~ 17x band`. The margin is not threshold-adjacent, so a ~4e-4 relative
numerical perturbation cannot flip it.

### What it does NOT establish

1. **Content discrimination on the live path.** All seven invalid competitors share one
   value per record (first record `{0:0.51188, 1:0.51188, 2:0.470383, 3:0.51188, ...}`),
   i.e. the do-nothing baseline, because those actions changed different cells. The live
   test is "true action vs do nothing". Content isolation is the local hard-comparator
   result: `own=0.03769 < S_hard(same cells, +2)=0.509042 < untrained=0.532079`.
2. **Generalisation beyond `ft09`.** One env, one policy.
3. **Any external task outcome.** ARC levels completed = 0. This is a mechanism probe.

### Kill budget

**0 of 2 consumed.** FORM B did not fail to separate; it separated in both runs. The
pre-registered kill requires two consecutive *failures with a populated store and real
role structure*. Neither condition for a strike was met.

### Next falsification

1. Independent replication — a run that differs in **env id or action policy**, since
   run #6 varied neither (~$0.11).
2. A live content-isolating comparator, if the environment produces an action whose
   support overlaps the true action's with different content. Cannot be forced.
3. `origin/main` stays at `adcc24e`. Nothing promoted.
