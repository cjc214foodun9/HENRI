# UHR-03 VERDICT — FORM B exercised live; the verdict is BLOCKED by a guard/consumer mismatch

Kill-run #3 of 2. Instance `52301758` (RTX PRO 5000 Blackwell), carrier `1d53798`,
overlay `799,034,119 B` / sha256 `7557238908...` (verified), env pinned
`ft09-0d8bbf25`, steps 32, tau 0.3500, `min_norm` 1e-5.

Cost: credit `$15.1476`; **0 instances running**; `52161444`, `52289752`,
`52301758` all `exited/stopped` (independently re-read).

## 1. What the instruction asked, and what was found

The instruction assumed the C1 update block was skipped by one of three inner
guards (`_aid >= 0` / `not learning_frozen()` / `su3_field is not None`).
**None of the three was the cause.** The blocking conjunct was an OUTER
module-scope ancestor, and the update path now runs in both arms:

| conjunct | value | verdict |
|---|---|---|
| `EXTERNAL_OUTCOME_EFE` (line 177, module scope, **0** force-set sites) | ancestor `if` at 2925/2861 | **the cause** |
| `_aid >= 0` | `aid` 1 and 2, `aid_ge_0` True 16/16 | passes |
| `not learning_frozen()` | `learning_frozen` False 16/16 | passes |
| `su3_field is not None` | `su3_field_present` True 16/16 | passes |
| `updated` | **True 16/16** in both arms | `update_generator` RUNS |

`OBSERVED`. Hypotheses recorded as FALSIFIED so they are not retried: a
module-level constant froze `HENRI_ARC_ACTION_EFE` (AST scope: force-set 488 and
read 613 are BOTH inside `run()`, force-set first); a paired-protocol freeze list
blocked learning (`FREEZE|freeze` = 0 hits in the launcher).

FORM B is adopted behind `HENRI_UHR02_EXTERO_GATE` (default OFF, additive).

## 2. Preconditions

| id | gate | measured | verdict |
|---|---|---|---|
| G1 | store holds a learned, NON-TRIVIAL transition | update path `target_theta_norm = 3.245417`, `stalled=False`; read path `truth_gen_frobenius = 9.75e-07` | **SPLIT — see §3** |
| G2 | `guard_state.updated == True` | 16/16 both arms | **PASS** |
| G3 | `extero_info.status == "OK"` | 16/16 (BASELINE 0 records) | **PASS** |
| G4 | both arms exit 0 | `exit=0` / `exit=0` | **PASS** |
| G5 | default-path identity | BASELINE emitted no `phase820_extero_info` | **PASS** |
| G6 | reference carries CONTENT structure | `magnitude_only_risk = True` 16/16 | **FAIL** |

## 3. The two measured causes (both mine, both CPU-invisible)

### 3a. Guard/consumer mismatch: the read path sampled ONE grid cell

`recorded_transition_generators` guarded on `theta_a[action].norm()` — the
AGGREGATE over `[8192, 8]` — but RETURNED `store.lie_element(action, basis)[0]`:
**channel 0 of a channel-resolved parameter**. The channel axis is a grid-cell
axis: `encode_su3_color_field(...).reshape(-1, 3, 3)` maps cell `(r, c)` to
channel `16*r + c`. Channel 0 is therefore ONE cell, and on the pinned `ft09`
grid that cell did not move.

Measured on the REAL store class (local, `DERIVED`):

```
||theta_a[2]||          (aggregate, guarded)  = 0.750022
||theta_a[2][0]||       (channel 0, consumed) = 3.6971e-08      <- 7 orders lower
||theta_a[2][:256]||    (real cells)          = 0.750022
real channels with a live transition          = 56/256
||lie_element(2)[0]||_F (what the gate read)  = 5.2284e-08   ~= live 9.75e-07
```

`relative_group_element = 3.000001` is the proof: `|Tr(U_c^dag U_t)| = 3` is the
identity, so BOTH operands were the identity and every residual was exactly
`0.0`. A guard that tests one quantity while the consumer reads another is the
signature-proof defect: 16/16 records looked successful.

### 3b. C1 as printed is VACUOUS and must not be reported as a pass

`argmin_hits_truth` was `True` 15/15. In all 16 records every candidate delta was
identical (`delta_extero_all = {1: -0.0, 2: -0.0}`), so `min()` is a TIE and
every candidate "argmins to truth". `C1 = 100%` carries **no information**.
`C2 invalid_minus_own = 0.0`, `C3 delta_extero = -0.0` — both FAIL.

A tie is not a hit. Any future C1 must be gated on
`max(delta)-min(delta) > sampling_band(8192) = 2.4705e-03`.

### 3c. The reference has no content structure

`role_coherence = ||mean(roles, dim=0)||` measured `0.0037 - 0.0086`, against
`1/sqrt(8192) = 0.0110` for isotropic random unit blocks. The Zone C axiom
reference is **isotropic**, and `magnitude_only_risk = True` 16/16 says the
comparison carries no more information than `|Tr|` — i.e. FORM B collapses onto
FORM A. Measured on the real store class: channel-resolved FORM B with an
isotropic reference gives `self = 0.007510` vs `vs-other = 0.006624`
(**no separation**) against a band of `2.4705e-03`.

## 4. Verdict

**`BLOCKED_INFRASTRUCTURE`. Kill budget: 0 of 2 consumed.** C1/C2/C3 are NOT
computable: G6 failed and the truth operand was the identity, so the store had
"a populated aggregate" but nothing to compare at the read channel.

`BLOCKED` does not count toward the pre-registered kill criterion, which
requires two consecutive runs with **a populated store AND real role structure**.

## 5. The derived conclusion — the repair is the reference, not the threshold

The domain fix was necessary and is now correct in form (the gate reads the
RELATIVE group element `|Tr(U_c^dag U_t)|`, not the option's own magnitude). But
the measurement shows the remaining gap is **content in the reference**, exactly
as pre-registered:

- with an isotropic reference the residual is a function of `|Tr|` only
  (`magnitude_only_risk = True`), and three different candidates score
  `6.6e-3 / 7.5e-3 / -0.0` — no discrimination;
- `tau` is not the lever: every value of `tau` in `[0, 1]` leaves all candidates
  on the same side when the deltas are tied;
- therefore the pre-registered conclusion holds: **the repair is a
  content-bearing baseplate — real role-filler assignments in the reference,
  not a threshold.**

## 6. Durable results (independent of this verdict)

1. **Attribution**: a dead flag is not a dead variable. `EXTERNAL_OUTCOME_EFE`
   has a live reader, a live consumer, and zero writers — reachable by
   construction, unreachable at runtime. Grep the WRITER count, not the symbol.
   Confirmed live at the exact line previously unreachable.
2. **Noise-floor calibration**: `min_norm = 1e-8` sat 320x BELOW the
   measurement's own noise floor (`3.2e-06` for `U U^dag` on float32). It is now
   `1e-5`. A floor below the noise floor cannot separate "no transition" from "a
   transition".
3. **Degeneracy boundary**: FORM B collapses onto FORM A below
   `||theta|| ~ 1e-01`; the residual scale is set by the transition angle, not by
   block count.
4. **Gate discipline**: a gate that PRINTS but does not STOP is not a gate. This
   class recurred four times this session (a pipe-masked dep gate, an
   env-var that never reached its consumer, an egress gate firing on manifest
   FORMAT instead of hashes, and a precondition line that printed
   `unexpected_sha=YES` and then ran anyway). Every precondition now asserts and
   exits.
5. **Diagnostic order**: a nested guard chain whose inner branch writes
   UNCONDITIONALLY can be attributed from telemetry alone — an all-`None` field
   proves the enclosing gate never ran.

## 7. Next falsification (fully specified)

1. Read the transition at the CHANNEL level (the 64 channels with the largest
   observed transition norm, deterministic), not at channel 0.
2. Gate C1 on candidate spread `> sampling_band`, so a tie can never be scored
   as a hit.
3. Require `magnitude_only_risk == False` on at least one record, else
   `BLOCKED_ROLE_STRUCTURE_ABSENT` (G6).
4. Re-run the SAME paired A/B. Kill unchanged: two consecutive runs satisfying
   G1-G6 that fail both C1 and C2 ⇒ `FALSIFIED`, and the repair is the
   content-bearing baseplate.

## 4. THE DOMAIN IS THE PER-CELL FIELD, NOT ONE CHANNEL (measured locally, 2026-09-23)

This is a FOURTH finding, and it is a DOMAIN defect, not a threshold defect. It
was measured offline at zero GPU cost, so it is not a post-mortem.

### 4a. Setup the finding rests on

`chromodynamic_grounding.encode_su3_color_field` (lines 99-118) maps an ARC frame
to `[B,H,W,3,3]` with `theta = one_hot(grid) @ projection` per cell: it is a
LOCAL per-cell map. Therefore one channel IS one grid cell, and the
channel-resolved `theta_a` carries one su(3) element per cell.

### 4b. Measured per-action support

Local rehearsal: `ActionOutcomeGeneratorStore(num_actions=4, num_channels=256,
lr=0.1)`, 16 updates per action, action-localized edits, production
`update_generator` / `lie_element` / `relative_displacement` calls only.

| action | top channel | live channels (>1e-5) | live range |
|---|---:|---:|---|
| 0 | 47 | 10 / 256 | [2..47] |
| 1 | 66 | 10 / 256 | [66..111] |
| 2 | 159 | 10 / 256 | [130..175] |
| 3 | 202 | 10 / 256 | [194..239] |

Channels live for ALL four actions: **0**.

Each action's learned transition lives ONLY in the cells that action changed. So
at any single fixed channel the admissible population is `n <= 1`:

| channel | n_admissible |
|---|---:|
| 0 | 0 |
| 47 | 1 |
| 66 | 1 |
| 159 | 1 |
| 202 | 1 |

With `n = 1` there is no `_others`, so `argmin_hits_truth` (C1) and
`invalid_minus_own` (C2) are UNCOMPUTABLE. The single-channel domain cannot
support the pre-registered criteria, independent of how well the gate works.

### 4c. The gate itself DOES discriminate when the domain is right

Identity candidate versus a real recorded transition, at each action's own
channel:

| action | channel | delta_pred | magnitude_only_risk |
|---|---:|---:|---|
| 0 | 47 | 0.135347 | False |
| 1 | 66 | 0.135347 | False |
| 2 | 159 | 0.135347 | False |
| 3 | 202 | 0.135347 | False |

against `sampling_band(8192) = 2.4705e-03`. That is ~55x band, with the
magnitude-only flag CLEAR. The mechanism separates; the domain does not.

### 4d. The reference is also read too LATE

The FORM B read (~line 3125) happens AFTER `update_generator` (~line 3069), and
`update_generator` folds the CURRENT observed transition into `theta_a` with
weight `lr` (`lr = 0.1`). The reference is therefore
`0.9 * EMA_{t-1} + 0.1 * target_t`: the step under test contaminates 10% of the
reference it is compared against.

Ordering receipt, measured: `update_generator` does NOT mutate its operands.
`max |U_t_after - U_t_before| = 0.000e+00`; its only write is
`self.theta_a[action].copy_(...)` at lines 81-84. So the fault is not mutation --
it is that the reference is read too late to be "recorded" in the intended sense.

## 5. CORRECTION (2026-09-23, same day, BEFORE any run #4 data) -- THE BASEPLATE
   REQUEST IS RETRACTED

Section 3c read `magnitude_only_risk = True` 16/16 as "the reference carries no
content structure" and asked for a **content-bearing baseplate**. That reading is
WRONG. The request is withdrawn. Both claims below are measured, at zero GPU cost,
with the production `exteroceptive_residual_vs_recorded`.

### 5a. What the flag actually measures

Definition (uhr02_exteroceptive_gate.py:520):
`magnitude_only_risk = bool(abs(delta_pred - delta_state) < band)`
with `delta_pred = delta(n(Ad(U_c) R), n(Ad(U_t) R))` and
`delta_state = delta(n(Ad(U_c) R), R)`.

If the recorded transition is the IDENTITY then `Ad(U_t) R == R`, so
`delta_pred == delta_state` **exactly**, the difference collapses to float noise,
and the flag fires regardless of role content. That is an algebraic identity, not
an empirical tendency.

### 5b. Controls (one variable changed at a time)

| transition | roles | delta_pred | delta_state | abs(dp-ds) | mag_only |
|---|---|---:|---:|---:|---|
| real | isotropic | 0.002016 | 0.180507 | 1.785e-01 | **False** |
| real | clustered | 0.001580 | 0.152202 | 1.506e-01 | **False** |
| identity | isotropic | 0.180507 | 0.180507 | 2.693e-08 | **True** |
| identity | clustered | 0.152202 | 0.152202 | 2.980e-08 | **True** |
| self-comparison (cand == ref) | isotropic | -2.700e-07 | - | - | False |

- Flag INVARIANT to role content (identical for isotropic and clustered roles at
  BOTH transitions): **True**.
- Flag TRACKS transition triviality (real -> False, identity -> True): **True**.

### 5c. Calibrated triviality threshold (replaces a hardcoded constant)

Sweep at fixed isotropic roles, transition angle scaled:

| scale | abs(Tr(U_c^dag U_t)) | delta_pred | abs(dp-ds) | mag_only |
|---|---:|---:|---:|---|
| 0 | 2.472475 | 0.180507 | 2.69e-08 | True |
| 1e-06 | 2.472476 | 0.180507 | 2.71e-07 | True |
| 1e-04 | 2.472566 | 0.180479 | 2.82e-05 | True |
| 1e-03 | 2.473392 | 0.180223 | 2.84e-04 | True |
| 1e-02 | 2.481618 | 0.177674 | 2.83e-03 | **False** |
| 5e-02 | 2.517503 | 0.166456 | 1.41e-02 | False |
| 2e-01 | 2.641893 | 0.126321 | 5.42e-02 | False |
| 1e+00 | 2.994612 | 0.002016 | 1.78e-01 | False |

So the flag is a TRIVIALITY detector with a measured knee between 1e-03 and 1e-02
on the transition scale. The runner's hardcoded `0.0707` Frobenius constant should
be replaced by this measured relation; `mag_only` itself is the ready-made signal.

### 5d. FORM B does NOT need a content-bearing reference

`experiments/verification/uhr02_domain_control.py`, run on this host:

    ISOTROPIC  FORM A sep=1.348e-04 NO-SEP   FORM B sep=3.638e-01 SEP
    STRUCTURED FORM A sep=1.565e-01 SEP      FORM B sep=4.657e-01 SEP
    tau band = (0.004355, 0.352165)   blueprint 0.35 lies in band

FORM B separates with **isotropic** roles at 147x the sampling band
(2.4705e-03). The role CONTENT is not the binding constraint. Run #3's G6
"failure" was the identity collapse (already measured: `relative_group_element =
3.000001`, truth `||H|| = 9.75e-07`), i.e. **the same defect** as defect 1, seen
through a second lens -- not a second defect.

### 5e. What remains, and it is NOT the baseplate

The binding constraint is the **DOMAIN** (section 4): at any single fixed channel
the admissible population is `n <= 1`, so C1/C2 are uncomputable there. The
repair is a SUPPORT-POOLED residual over the cells an action actually changed.
That IS load-bearing and IS unimplemented.

TRAP RECORDED so it is not walked into: a first attempt at the support-pooled
residual was **tautological**. When the reference for `a2 == aid` is built from the
same action's generators as the candidate, `U_c^dag U_t == I` by construction, so
`delta_pred = -2.700e-14` and "argmin picks the truth" is true BY DEFINITION.
Measured: `cand == field(AID)` elementwise = True, `|Tr| = 2.99998`. The pooled
residual must compare a LEARNED candidate against the OBSERVED displacement (as
the runner already does for `_truth`), never an action against itself.
