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
