# UHR-03 PRE-REGISTRATION — guard attribution + FORM B in the paired A/B

Written BEFORE any GPU spend. Criteria are frozen here so the verdict cannot
drift toward whatever the run happens to produce.

Carrier branch: `carrier/uhr-01-homologous-representation`
Blueprint: `HENRI-SPEC-2026-CAUSAL-REALITY-V1` section 3.2
Evidence labels: OBSERVED / DERIVED / INFERRED / HYPOTHESIS / FALSIFIED / BLOCKED

---

## 1. Attribution (already DONE locally, before the run)

The instruction asked which of three inner guards silently skipped
`update_generator` at `production_arc_run.py:3001`:
`_aid >= 0` / `not learning_frozen()` / `su3_field is not None`.

**Result: NONE of the three.** The blocking conjunct is an OUTER ancestor,
`if EXTERNAL_OUTCOME_EFE:` (originally line 2861; 2925 after the UHR-02
patches), read ONCE at module scope:

```text
 177| ind=0 | EXTERNAL_OUTCOME_EFE = os.environ.get("EXTERNAL_OUTCOME_EFE","0") == "1"
2925| ind=12| if EXTERNAL_OUTCOME_EFE:                 <- ancestor of the whole block
      os.environ["EXTERNAL_OUTCOME_EFE"] force-set sites in production_arc_run.py: 0
```

DERIVED (causal, from measured telemetry rather than a static read): inside the
gate, line 3067 `if stationarity_thermostat is not None:` merges `_tinfo` into
`p820_update_info` **unconditionally**, and `run.log` (both arms) prints
`[phase820] action-outcome store + thermostat armed (actions=8)`. Therefore a
`None` value at every step is possible ONLY if the gate never executed. Both
arms emitted `phase820_update_info = None` at all 8 steps.

OBSERVED rule-outs for the three NAMED guards:
- `su3_field is not None` PASSES: the fiber block (gated on it) printed
  `[fiber] un-collapsed 1 -> 2 admissible actions` 8 times per arm.
- `not learning_frozen()` PASSES: `HENRI_FREEZE_LEARNING` appears **0 times** in
  `uhr01_remote_ab.sh`; `learning_frozen()` returns `os.environ.get(...,"0")=="1"`.
- `_aid >= 0` UNREACHABLE, so unfalsifiable as the cause.

FALSIFIED hypothesis (recorded so it is not retried): "a module-level flag
constant froze `HENRI_ARC_ACTION_EFE` before `run()` force-set it". AST scope
puts both the force-set (488) and the read (613) inside `run()`, force-set
first. `HENRI_ARC_TARGET_GROUNDING` (500 / 634) is identical.

## 2. Preconditions (checked BEFORE the verdict; failure => BLOCKED_INFRASTRUCTURE)

**AMENDMENT 1 (2026-09-23, recorded BEFORE any data exists).** The step count is
raised `8 -> 16`. Reason: `--steps 8` produced only 8 records in UHR-01 and C1 is
conditioned on `n_recorded >= 2`, so 8 steps leaves too little room for the store
to accumulate more than one recorded transition. Criterion thresholds, tau, and
the kill rule are UNCHANGED; only the sample size moves. No data from either arm
of UHR-03 existed when this amendment was written (kill-run #1 died at
initialisation, line 722, before the step loop).

**AMENDMENT 2 (2026-09-23, before any data).** The common env block is carried
INSIDE the remote heredoc rather than as an ssh argv string. Measured defect in
kill-run #1: ssh joins its argv into one string which the remote shell re-splits,
so `COMMON="$5"` captured only the FIRST token (`HENRI_ARC_SAGNAC_VETO=1`).
`HENRI_OFFLINE_DIAG` therefore never reached the interpreter, `dsn` fell through
to `resolve_zone_c_dsn()`, and both arms raised at line 722
(`psycopg OperationalError ... 127.0.0.1:5434`). An env var that does not reach
its consumer is a mechanism that does not exist. A new `ENV_GATE` asserts every
flag inside the interpreter and fails closed before the arm starts.

**AMENDMENT 4 (2026-09-23, factual record — not a criterion change).** Record the
ACTUAL step count used in each kill-run, and freeze it for #4.

| run | steps | why |
|---|---|---|
| #1 | 8 | initial value (environment was static; irrelevant to the outcome) |
| #2 | 32 | EMA convergence: `theta_a` is an EMA at `lr=0.1`, so ~20+ updates per action are needed to push `\|\|theta\|\|` clear of the measured FORM-B collapse boundary `0.05` |
| #3 | 32 | identical to #2 by design (only the gate flag may differ) |
| #4 | 32 | frozen here |

Audit note: Amendment 1 wrote "8 -> 16" and the launchers then used 32. The
verdict for each run must cite the step count that ACTUALLY ran, so the audit
chain cannot carry an ambiguity of the same class this session was spent
eliminating. `32` is the frozen value from #2 onward; #1's `8` is recorded as
superseded and is not comparable.

**ADMISSIBILITY NOTICE — read before citing Amendments 5 and 6.**

Kill-run #3 was judged against the criteria frozen BEFORE it: Amendments 1-3
(`tau = 0.3500`, `min_norm = 1e-5`, env pinned `ft09-0d8bbf25`, C1/C2/C3 as
written in section 3, G1-G5). Run #3's result in `uhr03_verdict.md` cites ONLY
those.

Amendments 5 and 6 below were written AFTER run #3 was reduced. They CHANGE the
mechanism (which channel is read) and the statistic (C1 gains a spread
precondition). They are therefore **post-hoc with respect to run #3** and are
declared here as a NEW PRE-REGISTRATION governing kill-run #4 ONLY. Run #3 is NOT
re-judged under them. This distinction is recorded because the failure mode it
prevents — moving the goalposts after seeing the data — is the same class of
defect this session spent its time eliminating.

**AMENDMENT 5 (2026-09-23, post-hoc; governs run #4 only).** The candidate,
the reference, and the invalid population are read at ONE channel of the
channel-resolved store: the strongest-transition channel for the acting action
(`transition_channel`, argmax norm, ties by index). Reason: measured in run #3,
`lie_element(a)[0]` is a SINGLE grid cell, and on the pinned `ft09` grid that
cell did not move, so the candidate AND the reference were both the identity
(`relative_group_element = 3.000000 = |Tr(I)|`) and every residual was exactly
`0.0` while 16/16 records reported `status = "OK"`.

**AMENDMENT 6 (2026-09-23, post-hoc; governs run #4 only).** C1 gains a vacuity
precondition: a record counts toward C1 only when `spread_above_band == True`, i.e.
`max(delta_extero_all) - min(delta_extero_all) > sampling_band(8192) = 2.4705e-03`.
Run #3's `argmin_hits_truth = True 15/15` was a TIE artifact (all deltas `0.0`)
and is recorded as NOT a C1 pass.

(a) **The environment must be PINNED to a moving one.** The store is trained from
the OBSERVED frame transition, not from demos, so kill-run #1 read an identity.
`HENRI_SINGLE_ENV=ft09` is added to the COMMON block (measured 8/8 moving in
UHR-01); without it the API's first-listed env is used and it rotated to the
static `lp85`.

(b) **The admissibility floor is raised `1e-8 -> 1e-5`** in
`recorded_transition_generators`. The old floor sat BELOW the measurement's own
noise floor: `_matrix_log_eig` returns ~`3.2e-06` for `delta_U = U U^dag` on
float32 and exactly `0.0` only for an exact identity, so `1e-8` admitted a pure
identity's residue as a "recorded transition". `1e-5` is ~3x the noise floor and
5 orders below the smallest genuinely learned transition (`||H||=1e-03 ->
9.05e-02`). Below the new floor the gate returns UNAVAILABLE, which is `BLOCKED`,
never a negative result.

Both arms run with the SAME environment except the one experimental flag:

```text
  common : HENRI_ARC_SAGNAC_VETO=1, HENRI_MACRO_NUM_CHANNELS=1, HENRI_OFFLINE_DIAG=1,
           EXTERNAL_OUTCOME_EFE=1, HENRI_TRACE_UPDATE_GATES=1,
           --mode phase823_live_gauntlet --envs 1 --steps 8
  arm A  : HENRI_UHR02_EXTERO_GATE=0
  arm B  : HENRI_UHR02_EXTERO_GATE=1
```

`EXTERNAL_OUTCOME_EFE=1` is part of the COMMON block precisely because it is the
attributed cause of the empty store. It is not the experimental variable.

- **G1 store populated**: `sum(|theta_a|) > 0` by the final step, visible in
  `phase820_update_info.target_theta_norm`. If false => `BLOCKED_INFRASTRUCTURE`,
  NOT a negative result.
- **G2 update engaged**: `phase820_guard_state.updated == True` on >= 1 step.
  If false => `BLOCKED_INFRASTRUCTURE`.
- **G3 FORM B defined**: `phase820_extero_info.status == "OK"` on >= 1 step.
  `UNAVAILABLE_*` is `BLOCKED`, never "FORM B failed to separate".
- **G4 both arms exit 0**. Any nonzero exit => `BLOCKED_INFRASTRUCTURE`.
- **G5 C3 default-path identity**: with `HENRI_UHR02_EXTERO_GATE=0` and
  `HENRI_TRACE_UPDATE_GATES` unset, the emitted record set must be
  byte-identical to the pre-patch baseline except for the two added keys, which
  must be `None`.

## 3. The experimental criterion

τ is FROZEN at the blueprint value `0.3500` (`TAU_BLUEPRINT`). It is NOT tuned.

- **C1 (primary, content discrimination)**: with the candidate option HELD
  FIXED and the reference varied over every RECORDED transition, the residual
  minimum must fall on the action actually executed more often than chance:
  `argmin_hits_truth == True` on >= 50% of steps where `n_recorded >= 2`.
  Rationale: if FORM B were magnitude-only, all references would give the same
  residual and the argmin would be arbitrary. This is the discriminating control
  from `henri-co-scientist-rigor` (vary the reference, hold the operator fixed).
- **C2 (population separation)**: over steps with `status == "OK"`,
  `invalid_minus_own > 0` on >= 50% of steps (the true reference is not worse
  than every alternative).
- **C3 (domain still fixed)**: `delta_extero` must leave the degenerate
  `{0.0}` set. A constant `delta_extero` reproduces the UHR-01 failure mode and
  is `FALSIFIED` regardless of C1/C2.
- **C4 (no fail-open on identity)**: `delta_identity`, the do-nothing candidate,
  is reported. Descriptive only -- it is NOT a pass/fail gate.

## 4. Kill criterion (pre-registered, user-specified)

- **One failing run is not a kill.** Two CONSECUTIVE live runs that satisfy
  every precondition G1-G5 and satisfy neither C1 nor C2 => `FALSIFIED`.
- On `FALSIFIED`, the repair is a **content-bearing baseplate** -- real
  role-filler assignments in the reference -- **not** a threshold change.
  Recorded in the evolution-loop reject log so it is never retried blind.
- A `BLOCKED_INFRASTRUCTURE` outcome consumes NO kill budget.

## 5. Comparability constraint (carried from UHR-01)

`USE_ZONE_C_AXIOMS` stays UNSET in both arms, so the reference is the per-frame
residual -- the same object as in the UHR-01 run whose `delta_axiom` population
was recorded (`0.997175 .. 0.999582`). A different reference would make the two
experiments non-comparable.

Consequence recorded honestly: `boundary_batch[0]` is NOT guaranteed unit-norm
or structured. `phase820_extero_info.role_coherence` IS emitted so the kill
verdict is decidable between "domain still wrong" and "reference is isotropic".
Role structure is MEASURED, not assumed.

## 6. What this run cannot establish

- It cannot promote `main`. `origin/main` stays at `adcc24e` regardless.
- It cannot establish task progress: `HENRI_OFFLINE_DIAG=1` (`offline://surrogate`)
  makes the run DIAGNOSTIC-ONLY and never score-eligible.
- It cannot substitute for CUDA verification of the FORM B math; the CPU probes
  are `DERIVED`, and the live run is the `OBSERVED` layer.

**AMENDMENT 7 (2026-09-23, POST-HOC with respect to kill-run #3; governs run #4
onwards only).** Recorded BEFORE run #4 data exists. Three measured findings
force a DOMAIN change -- not a threshold change.

**A7.1 -- the comparison domain is the per-cell field, not one channel.**
`encode_su3_color_field` is a LOCAL per-cell map, so channel == grid cell, and
`theta_a` is one su(3) element per cell. Measured support (4 actions x 16
updates, 256 cells): top channels 47 / 66 / 159 / 202, each with 10/256 live
channels and DISJOINT ranges; channels live for ALL actions = 0. At any single
fixed channel the admissible population is `n <= 1` (ch 0 -> 0; ch 47, 66, 159,
202 -> 1), so `_others` is empty and C1/C2 are UNCOMPUTABLE. C1/C2 must be
computed over the action's SUPPORT (a pooled per-cell residual), not at one
channel.

**A7.2 -- the reference must exclude the step under test.** The FORM B read
(~line 3125) runs after `update_generator` (~line 3069), which folds the CURRENT
transition into `theta_a` with weight `lr = 0.1`. The reference is therefore
`0.9*EMA_{t-1} + 0.1*target_t`, i.e. contaminated by the step under test.
Ordering receipt (measured): `update_generator` does not mutate its operands
(`max|U_t_after - U_t_before| = 0.000e+00`); the fault is read TIMING, so the
reference must come from a PRE-UPDATE snapshot of `theta_a`.

**A7.3 -- admissibility preconditions for run #4.** Before any criterion is
read, the run must report: `truth_operand_untouched` (not `None`),
`nontrivial_transition = True`, `spread_above_band = True`, `n_recorded >= 2`,
`channel != 0`, and role coherence at or above the isotropic baseline. Any
failure is `BLOCKED_INFRASTRUCTURE` and consumes NO kill budget.

**STATUS: A7.1 and A7.2 are LOAD-BEARING and are NOT implemented.** Run #4 must
not be launched until they are. The pre-registered kill rule is unchanged and
still holds 0 of 2 budget.

**Baseline note (Q1, measured).** The live emit is
`float(_roles.mean(dim=0).norm())` (line 3229). Its isotropic baseline, measured
numerically over 20 draws at K=8192, d=8, is **0.011129** (min 0.007906, max
0.017516), matching the analytic `1/sqrt(K) = 0.011049`. `sqrt(2/(pi*K)) =
0.008815` is the baseline for mean PAIRWISE |cos|, a DIFFERENT statistic
(measured 0.291379 for unit rows, since pairwise |cos| scales as 1/sqrt(d), not
1/sqrt(K)). Run #3's observed 0.0037..0.0086 sits AT or BELOW the low end of the
isotropic band, so the roles are spread more evenly than random.

**AMENDMENT 8 (2026-09-23, POST-HOC wrt run #3; governs run #4 onwards).**

**A8.1 -- the baseplate requirement is WITHDRAWN.** Amendment 7's G6 reading was
wrong: `magnitude_only_risk` reports TRIVIALITY OF THE RECORDED TRANSITION, not
absent role content (see uhr03_verdict.md section 5; controls vary one variable at
a time; the flag is invariant to role content and tracks transition triviality).
FORM B separates with isotropic roles at 3.638e-01 = 147x band, measured by the
module's own calibration harness `uhr02_domain_control.py`. **No content-bearing
baseplate is required and none should be built.**

**A8.2 -- replace the hardcoded triviality constant with the measured knee.**
The runner's `nontrivial_transition` uses a fixed Frobenius `> 0.0707`. Measured,
the flag's knee lies between 1e-03 and 1e-02 on the transition scale. Report BOTH
`mag_only` (the gate's own signal) and the Frobenius magnitude; do not gate on the
hardcoded constant alone.

**A8.3 -- admissibility for run #4 (supersedes A7.3).** Before ANY criterion is
read, the run must emit and the reducer must assert:
`truth_operand_untouched is True`; `nontrivial_transition is True`;
`magnitude_only_risk is False`; `n_recorded >= 2`; `channel != 0`;
`role_coherence` at or above the measured isotropic baseline (0.011129 at K=8192).
Any failure => `BLOCKED_INFRASTRUCTURE`, consuming NO kill budget.

**A8.4 -- C1 is a MARGIN condition, not a spread condition.**
`delta_spread > band` proves only that the deltas are non-degenerate. C1 must be
`delta_second_min - delta_min > sampling_band`, i.e. genuine DISCRIMINATION.
Run #3's `argmin_hits_truth = True 15/15` was a tie artifact and is NOT a C1 pass.

**A8.5 -- the load-bearing item is the SUPPORT-POOLED domain (A7.1), not the
baseplate.** Still UNIMPLEMENTED. A tautological attempt is recorded in
uhr03_verdict.md section 5e so it is not repeated.

**AMENDMENT 9 (2026-09-23, recorded BEFORE kill-run #6 data). Reproducibility pass.**

Sentinel for content-based idempotence (this file already contains an unrelated
Amendment 8 that governs run #4, so a LABEL check is not a safe guard):
RUN6_REPRODUCIBILITY_PREREG_SENTINEL

Kill-run #5 returned C1 = 16/16 and C2 = 16/16 with `margin mean = +0.042074`
(range `+0.041497 .. +0.042304`, i.e. 16.8x .. 17.1x band). That is one live pass:
`N = 1`.

The pre-registered kill rule requires two CONSECUTIVE failures. Symmetrically, a
promotion claim requires a REPRODUCED pass. Run #6 is frozen here, before any of its
data exists:

- Same carrier, same pin (`HENRI_SINGLE_ENV=ft09`), same `STEPS=32`, same criteria.
- Same two arms, differing ONLY in `HENRI_UHR02_EXTERO_GATE`.
- Judged by the identical G1-G6 / C1-C3 criteria at the same `tau = 0.3500` and the
  same `band = sampling_band(8192) = 2.470529e-03`.

Outcomes and their meaning, fixed now so the interpretation cannot be chosen later:

| run #6 result | interpretation |
|---|---|
| C1 and C2 both 100% | PASS REPRODUCED. Two independent live runs agree. |
| C1 or C2 in [50%, 100%) | PASS REPRODUCED WEAKLY; margin not stable across runs. |
| C1 or C2 below 50% | **Kill strike 1 of 2.** The run #5 pass was a draw artifact. |

One comparability condition, also fixed now: run #6 must report the same
**identity-only competitor set** (every invalid action sharing one value) for its C1/C2
to be comparable with run #5. If run #6 produces a competitor set containing a
non-identity value, report it separately as a CONTENT-OBSERVING run and do NOT pool it
with run #5. Run #5's live competitor set was identity-only, so run #5 evidences action
IDENTIFICATION, not content discrimination.

**Provenance note (twice-repeated defect in my own tooling).** Two earlier attempts to
record a run #6 pre-registration used LABEL guards — first `if "AMENDMENT 7" not in
text`, then `if "AMENDMENT 8" not in text`. Both labels were already occupied by
unrelated post-hoc notices governing run #4, so both guards did nothing, and each
printed "already present": a FALSE claim, twice. This amendment is guarded by the
content marker above and verified by substring search on the COMMITTED blob, not on the
working tree. **Guard on content, never on a label that a later writer may reuse.**

**AMENDMENT 10 (2026-09-23, POST-HOC characterisation of kill-run #6).**

Run #6 satisfied Amendment 9's letter ("C1 and C2 both 100% -> PASS REPRODUCED") but
**not its spirit**. Amendment 9's row assumed an independent sample. Measured, run #6
is a *numerical robustness check on the same trajectory*:

| quantity | run #5 | run #6 | same? |
|---|---|---|---|
| env | `ft09-0d8bbf25` | `ft09-0d8bbf25` | yes |
| action sequence | `ACTION2/ACTION1` x16 | `ACTION2/ACTION1` x16 | yes |
| `changed_cells` per step | 4 | 4 | yes |
| `channels` per record | `[4092..4095]` | `[4092..4095]` | yes |
| `target_theta_norm` | `3.245417356491089` | identical, all 38 records | yes |
| seed | **none** (`_launch.sh` passes no seed) | none | n/a |

Differences are float-level only: `own_min` `0.44926` vs `0.449444`; `margin_sum`
`0.673191853` vs `0.673361203`; **1 of 16** records bitwise identical. The relative
perturbation is ~4e-4, i.e. GPU kernel reduction-order noise.

**What this does and does not buy.**

- It DOES show the verdict is robust: a ~4e-4 relative perturbation does not move
  `own_is_min` in any of 16 records, because the margin is ~17x band. The margin is
  not sitting near a threshold.
- It does NOT show the result replicates across independent draws. Same env, same
  deterministic policy, same channel set means the trajectory is repeated, not resampled.

**A future independent draw requires a different env or a different action policy**,
neither of which run #6 varied. Governing rule from here: a run only counts as an
independent replication if it differs in env id or action sequence; otherwise it is
labelled a robustness check. This is recorded POST-HOC with respect to run #6, and
governs run #7 onward.

**Residual anomaly, recorded rather than deleted.** My first comparison of the two
archives printed identical margin means (`0.042085`) for both, which would have implied
one payload read twice. Explicit per-archive enumeration by filename + sha256 shows they
differ (`7d3331db6c301878` vs `265fd9363da4d2d8`). The first comparison was buggy; the
explicit enumeration is authoritative. The buggy reading is left in the session record.

**Storage cost flag (needs human approval to act).** Three stopped disks hold 60 GB each:
`52161444` (`henri-v2-fullscale`), `52289752`, `52301758` (`henri-v2-uhr03`).
Burn = `180 GB * 0.2220 / 30 = $1.3320/day` against credit `$14.6292` (~11 days).
Destruction is irreversible and is NOT performed without approval.

**AMENDMENT 11 (2026-09-23). Independent recomputation of the live C1, and the
single-channel vs pooled distinction.**

The live C1 was previously read from the helper's own `own_is_min` field. It is now
recomputed from the raw population `delta_pooled_all`, without trusting that flag:

```
run #5 RFSS: own < min(delta_pooled_all minus own)  ->  16/16 records
run #6 RFSS: same                                    ->  16/16 records
```

Both runs agree. The subject is the strict argmin of the full 8-action population in
every scored record.

**The single-channel and pooled paths differ, and they must not be confused.**
`phase820_extero_info.magnitude_only_risk` is `True` on some records while the pooled
path still ranks correctly. That is expected and is not a contradiction:

- `magnitude_only_risk` describes **FORM B at ONE channel** (`channel: 0`), i.e. one
  grid cell. A single cell is frequently identity, so a candidate-vs-state comparison
  there is magnitude-only.
- The **pooled** statistic averages over the cells the observation moved
  (`n_channels = 4` at `[4092..4095]`), where the transition is non-trivial.

So a `True` on the single-channel flag is a statement about channel 0, not about the
pooled verdict. Any future reader must not use `magnitude_only_risk` to impeach the
pooled C1.

**Role structure, measured not assumed.** The kill rule requires "a populated store
AND real role structure". Measured `role_coherence` (the emitted statistic
`float(roles.mean(dim=0).norm())`): run #5 `0.003661 .. 0.008903`, run #6
`0.003661 .. 0.008722`. The isotropic baseline for THIS statistic, measured
numerically at `K=8192, d=8`, is `0.011129` (and `1/sqrt(K) = 0.011049`). Observed
values sit BELOW isotropic, i.e. the axiom roles are more evenly spread than random —
a design property, not absence of content. Note `sqrt(2/(pi*K)) = 0.008815` is the
baseline for a DIFFERENT statistic (mean pairwise |cos|, measured `0.291`) and must not
be substituted here.

**`|Tr(U_c^dag U_t)|`** measured in `{2.973726, 2.978709, 3.000001}`. The value
`3.000001` is `|Tr(I)|`, so at least one record's reference is the identity — the
`delta_extero = -0.0` family from run #3. The pooled path still ranks correctly there
because it also reads the channels that did move.
