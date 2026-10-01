# HENRI-ARCH-2026-CRITICAL-DIRECTIVE-V1 — Execution Record

**Status:** three directives executed. Each outcome is labelled by evidence class.
**Baseline:** `72f20fe` (+ this commit). The directive document cites `HEAD e3e6729`;
that is an **earlier** commit in this repository (`Pipeline: portable flat(), measure/render
split...`), so the document's audit baseline is **stale, not fabricated**.
**Date:** 2026-09-30.

## 0. Verification of the document's targets (before acting)

Every module the document names **exists**. An earlier draft of this record asserted
otherwise; that assertion was wrong and is corrected here.

| identifier | status |
|---|---|
| `e3e6729` | real commit object (earlier pipeline phase) |
| `recursive_dual_edmd.py`, `wave_jepa.py` | present |
| `henri_hopfield_egress.py`, `hopfield_cleanup.py` | present |
| `henri_calibrated_action_head.py` | present (Stiefel-ridge action proctor) |
| `henri_scene_binder.py`, `henri_operator_router.py` | present |
| `arc_scorecard_delta.py`, `o_vsa_ingress_tokenizer.py`, `arc_task_functor.py` | present |

**One targeting error in the document.** Directive 2 says continuous waves terminate in
`henri_calibrated_action_head.py` "using Modern Hopfield retrieval at beta*=26.10".
`henri_calibrated_action_head.py` contains **no beta/temperature constant at all** — it is
a Stiefel-ridge proctor. The Hopfield beta lives in `hopfield_cleanup.py` (core) and
`henri_hopfield_egress.py` (`CanonicalCodebookEgress`). The beta was changed at the real
site. Both the document's stated module and the real one are recorded here.

## 1. Directive 1 — do not spend resources on linear EDMD

**DONE. No compute was reserved or spent; the instance is `exited`.**

`recursive_dual_edmd.py` now carries a **DEPRECATED FOR STATE-TRANSITION PREDICTION**
header with the measured bound and a deliberately narrow scope:

- **DEPRECATED:** this module as `WaveJEPA`'s transition predictor (`wave_jepa.py:61`).
- **NOT DEPRECATED:** `CoupledRecursiveDualEDMD` (still consumed by the Phase 8.34 benchmark).
- **NOT DELETED:** the attribution ladder in `tier2_measured_gate.py` — that code is what
  *answered* the capacity-vs-architecture question. Evidence, not debt.

The bound (`OBSERVED`, committed at `72f20fe`): best LINEAR 3-step cosine **0.280163**
(`d=65536`) / **0.232242** (`d=512`) against a **0.92** gate, invariant over
`lambda` in `[1e-5, 1e-2]`. It carries **no rank bottleneck**, so it is an upper bound for
the linear class — rank, learned basis, and `lambda_forget` sweeps cannot reach 0.92.

## 2. Directive 2 — typed egress at beta* = 26.10

**MEASURED. 26.10 IS strictly better than the sealed 8.0. `ADOPT_26.10_MEASURED` — at
`d=512`, which is BELOW this constant's named promotion scale. The code change is therefore
`PROVISIONAL`.**

**Promotion status: `PROVISIONAL — pending M=10,000 / D=65,536`.**

The prior receipt `experiments/verification/hopfield_beta_calibration.json` states the
governance rule for this exact constant:

> "Changing the sealed constant requires replicating the M=10,000 / D=65,536 capacity
> contract and passing the receipt-pinned promotion gate. This sweep deliberately does not."

This work changed the value on fixtures at `d=512`, `M=64` and `M=174`. That is direct
measurement with passing controls, but it is **not** the production capacity contract, so it
does not satisfy that gate. Precise governance scope: `beta` is **not** an entry in
`validate_seal_consistency.py` (`PAIRS`/`SCALAR_SPECS`), so no *registered* seal gate was
tripped — the requirement above is the one the constant's own prior receipt names.

A first revision of this record said `ADOPT_26.10_MEASURED` without disclosing the scale
gap. That is the same defect class as the convergence table that asserted a fired control
it could not show. The disclosure is now inline.

**The promotion gate CANNOT be closed by this harness — a fixture limitation, measured.**
The gate requires `M=10,000`. `egress_beta_gate.py`'s grid generator indexes every pattern
by `k % g`, so its unique-grid family caps at **174** (measured: a request for `M=256`
raised `size of tensor a (174) must match b (256)`, and the `M=174` request yielded exactly
174). The family grows only linearly in `g`, so no practical grid reaches 10,000. Closing
the gate therefore requires a **real production engram source** (waves encoded from the
actual codec or benchmark corpus), not an enlarged synthetic family. That fixture is **not
yet built**. An earlier revision of this paragraph asserted `--m 10000` as the fix; that was
a claim the artifact could not support, and it is recorded here rather than quietly dropped.

Consequently the GPU window instruction is corrected: run `--live` and
`tier2_measured_gate.py` as planned. Do **not** expect production-scale `beta` confirmation
from that window.

**The requirement is now machine-readable, not prose.** This section previously stated the
promotion requirement only as prose, which is bypassable: a session that reads this doc and
not the prior receipt would see no enforcement. That is the same failure
`henri_operator_promotion.py` was written to prevent for operator families — a supplied
directive once asked for a resonator family that had already been measured worse.

`henri_egress_promotion.py` (commit `83e1779`, tests
`tests/unit/test_egress_promotion.py`) makes it callable:

* `audit()` re-reads both β-gate receipts and verifies pinned digests, so the constants
  cannot silently drift. Digests are **LF-normalised** and derived from the git-stored
  blobs; the sibling gate hashes raw bytes and is therefore checkout-fragile on a host that
  rewrites line endings (this worktree warns about exactly that).
* `assert_promotion_allowed(M, d)` raises `PromotionGated` unless a measurement was taken
  at or above `M=10,000 / D=65,536`. Fail-closed on a missing receipt, a digest mismatch,
  or a sub-scale measurement.
* `is_provisional()` is `True` today, and `HopfieldTerminator.report()` carries the result
  under `promotion_gate`, so `PROVISIONAL` reaches a telemetry surface rather than living
  only in this document.

Verified from the committed code, not the working tree: a conforming `(10000, 65536)`
measurement **passes**, a sub-scale `(174, 512)` **raises**. The gate blocks what was
actually measured and still opens on a conforming one — a gate that can only fail is the
mirror of a gate that can only pass.

New instrument: `experiments/verification/egress_beta_gate.py`.
`CanonicalCodebookEgress` (`henri_hopfield_egress.py:60`) default changed `8.0 -> 26.10`
with full provenance in the docstring. The prior `sealed_beta_note` in
`henri_dream_compass.py`, which asserted the opposite, was **updated**, not left to
contradict the new receipt.

Instrument = **noise tolerance**: the largest `||noise||/||engram||` at which cleanup keeps
`clean_cos >= 0.90`. Real engrams: `HENRIVisionEncoder` structured grids.

| fixture | M/d | beta=8.0 | beta=26.10 | beta=64 |
|---|---|---|---|---|
| M=64 | 0.125 | 2.0 | **4.0** | 6.0 |
| M=174 | 0.340 | 1.5 | **4.0** | 4.0 |

Controls, all passing: **invariance** (argmax P@1 spread `0.000000` — proving the old
metric cannot depend on beta), **sensitivity** (`soft clean_cos` spread `0.61`),
**resolution** (5–6 distinct tolerances), **separability** (max off-diagonal cos `0.82`/
`0.90` — SEPARABLE).

### Why the prior receipt said the opposite — and why that was not wrong, just blind

`hopfield_beta_calibration.json` reported
`DOC_BETA_NOT_BETTER_THAN_SEALED__NEVER_STRICTLY_BETTER`, `unique_argmax: false`. Its own
`honest_limit` disclaimed the fixture: *"Synthetic random codebook... NOT recall over a real
embedding distribution."* On a random codebook all large betas saturate, so every
temperature ties.

**Two instrument failures, both structural:**

1. **Synthetic fixture** — cannot represent the real embedding distribution.
2. **Beta-invariant metric (the deeper defect).** `lexical_snap` takes
   `sim.argmax(dim=-1)` and never reads `self.beta`. More generally, softmax is monotonic:

   ```
   argmax(softmax(beta * sim))  ==  argmax(sim)   for EVERY beta > 0
   ```

   **Any P@1-by-argmax egress metric is beta-invariant by construction and can never
   calibrate a temperature.** This is why `unique_argmax: false` appeared: the quantity
   measured does not depend on the parameter being calibrated. The new instrument carries
   an explicit invariance control that asserts this flatness, so the blind spot cannot
   silently return.

### Scope limit (do not over-read)

**26.10 is NOT the optimum.** At M/d=0.125, `beta>=64` reaches tolerance **6.0** > 4.0, and
`beta=64/128/256` all beat 26.10. The measured claim is exactly: *strictly better than the
sealed 8.0, at both tested densities.* The directive's "empirically derived optimum" is
**not** supported; the directive constant is retained because it is mandated and now has
positive evidence, not because it is optimal.

## 3. Directive 3 — non-linear world model

**MEASURED, NEGATIVE at this scale — with the memorisation control firing, so it is NOT a
clean verdict.** New probe: `experiments/verification/tier2_nonlinear_probe.py`.

Fixture is shared with the linear harness **by importing it**, so the two arms are provably
the same distribution. Split is by **whole trajectory seed** (train `[11,22,33,44]`, test
`[55,66]`) to defeat mememorisation.

| steps | arm | train 3-step | test 3-step | mem gap | flag |
|---|---|---|---|---|---|
| 40 | mlp | 0.149 | 0.054 | +0.095 | |
| 40 | mlp+skip | 0.008 | −0.000 | +0.008 | |
| 400 | mlp | 0.313 | 0.125 | +0.188 | |
| 400 | mlp+skip | 0.337 | 0.168 | +0.170 | |
| 2000 | mlp | 0.490 | 0.209 | +0.280 | **MEMORISATION** |
| 2000 | mlp+skip | 0.403 | 0.216 | +0.187 | |
| 12000 | mlp | 0.495 | 0.197 | +0.298 | **MEMORISATION** |
| 12000 | mlp+skip | 0.445 | **0.226** | +0.219 | |

**Receipt provenance — every row is backed by a committed artifact.** An earlier revision of
this section cited these numbers with only the 12000-step receipt in version control; the
other three were on disk but **untracked**, so a future reader could not verify 6 of the 8
rows. All four are now committed and re-checked against this table at landing time
(fail-closed in `land_nonlin_receipts.sh`):

| steps | receipt |
|---|---|
| 40 | `receipts/tier2_nonlinear_cpu.json` |
| 400 | `receipts/tier2_nonlinear_cpu_400.json` |
| 2000 | `receipts/tier2_nonlinear_cpu_2000.json` |
| 12000 | `receipts/tier2_nonlinear_cpu_12000.json` |

All four were produced by the probe as it stands in this commit (verified: the probe is
byte-identical to `HEAD`), so they share one code path. Each carries
`verdict: NONLINEAR_KILLED_AT_THIS_SCALE`; the **memorisation** observation is a property of
the `mlp` arm's `mem_gap` column, computed and reported by the probe, not a separate verdict
string. Do not attribute a `MEMORISATION_GUARD_FIRED` verdict to these receipts — no such
string exists in them.

**Both arms are shown, including the rows where the guard FIRED.** An earlier revision of
this table listed only the `mlp+skip` rows (all below the 0.25 flag) and asserted in prose
that the memorisation control fired — an assertion its own table could not evidence. The
flagged rows are now visible: the plain-MLP arm crosses the threshold at 2000 steps
(`+0.280`) and again at 12000 (`+0.298`).

**The 400-step row is clean and must not be cited alone.** It is the single point where the
gap is smallest on the skip arm (`+0.170`); citing "400 steps shows no memorisation" would
be *true per-row and false about the ladder*, because the gap rises monotonically with
optimisation on both arms. This is the same fault as the withdrawn `CAPACITY` verdict: a
single favourable sample promoted to a general claim.

Linear reference on the same fixture: 3-step **0.1457** (metric validity OK).

**Three honest readings:**

1. **Non-linearity does beat the linear reference on the same fixture** (0.226 vs 0.146,
   `+0.080`) — consistent with the 0.280 bound covering the linear class only.
2. **It does not approach the 0.92 contract**, and plateaus (`0.216 -> 0.226` between 2k
   and 12k steps while loss keeps falling) — so more steps of THIS class will not close it.
3. **The memorisation guard FIRED on the plain-MLP arm** (`+0.280` at 2000, `+0.298` at
   12000), and the gap on the skip arm rises monotonically (`+0.008 → +0.170 → +0.187 →
   +0.219`). The test numbers are therefore **not clean generalisation results**. The
   fixture has too little trajectory diversity for a 512-hidden MLP.

**Therefore this is NOT "non-linearity is dead."** It is: *a flat MLP of this size, on this
narrow fixture, plateaus well below contract.* The directive's own standard —
**Hierarchical, Nested, Recursive** — rules out another flat one-shot map; a nested/recursive
predictor with multi-seed trajectory diversity is the untested branch. Stated as a
pre-registered hypothesis, not a capability claim.

## 4. What is NOT done, and is not silently skipped

- **G5 (TimescaleDB causal DAG)** — no database confirmed present in this environment. A
  blocked item, not a completed one.
- **Pillar 3 (Langevin thermalisation on Sagnac veto), Pillar 1 (Clifford fibre encoder)** —
  not implemented. No measurement exists for `Delta_Sagnac` rejection-to-creep coupling.
- **The document's unsourced figures** — `12.8 µs` Triton, `16/16` torus invertibility,
  `ARC 0.0% BLOCKED_NO_DEMONSTRATIONS`, `reward_mean 1.71e-6`, `Sagnac 0.35` — do **not**
  appear in any receipt produced here. They are recorded as **UNVERIFIED**, not adopted.
- **`--live` on GPU** — BLOCKED (instance `exited`; restart refused 6/6 on machine 143423).
  No GPU measurement is claimed anywhere in this record.
- **Mutual-information collapse** (`I(Psi;Y) -> 0`) — asserted by the document, **not
  measured**. The typed-egress change is justified by the beta measurement above, not by MI.

## 5. Threshold discipline

No threshold was relaxed at any point. `0.92 / 0.60 / 1e-5` are module constants,
pre-registered, and unchanged. No `tier2_measured_thresholds.json` exists.


### 3.3 Pillar 2 -- per-block non-linear slot mixer: NEGATIVE (empirical, NOT a proven ceiling)

Two structured arms are now measured on the identical fixture. Their failures have
DIFFERENT status and must not be merged.

| arm | mechanism | test 1-step | test 3-step | gap | params |
|---|---|---|---|---|---|
| linear (reference) | dual-form ridge | 0.315731 | 0.145739 | -- | -- |
| mlp (flat) | 512-2 MLP | 0.483063 | 0.196783 | +0.298111 | 787968 |
| mlp+identity-skip | 512-2 MLP + skip | 0.480529 | 0.226031 | +0.219416 | 787968 |
| resonator | rotor + mask + shift (all linear) | 0.223327 | 0.088508 | +0.004597 | 643 |
| mlp-per-block | resonator + per-block GELU mixer | 0.052690 | 0.056038 | +0.022602 | 1195 |

Receipt: `receipts/tier2_blockmix_cpu_12000.json`. d=512, nb=64, steps=12000, device cpu,
seeds train [11,22,33,44] / test [55,66], rc=1, verdict `NONLINEAR_KILLED_AT_THIS_SCALE`. The three arms
measured in the previous receipt reproduced their committed values EXACTLY
(0.196783, 0.226031, 0.088508).

Pre-registered condition, fixed BEFORE the run: `test3 > 0.226 AND gap < 0.25`. Measured
0.056038. **CONDITION FAILS.**

The distinction that must survive to the next session:

- The `resonator` failure is PROVEN, not merely measured. `R`, `Pi`, `T` are all linear,
  the update is a linear combination, and the exit `normalize` changes radius only while
  cosine reads direction. The composite cannot leave the linear class, so the 0.280/0.232
  ceiling applies to it.
- The `mlp-per-block` failure is EMPIRICAL ONLY. The arm was verified to leave the linear
  class (direction-proportionality 1.461e-01 off zero-init, vs 8.2e-08 for the linear arm),
  and the mixer was instantiated (+552 params, exactly as designed). Nothing here shows a
  ceiling. It scored 0.052690 on ONE step -- below the linear
  reference (0.315731) and below every other arm -- while fitting its own pairs
  worse than the resonator (loss 3.57e-03 vs
  3.01e-03). That is an optimisation failure on this fixture, not
  a class bound.


---

## 6. Pillar wiring audit -- VERIFIED, three findings (2026-10-01)

Earlier sections recorded Pillars 3 and 5 as "wiring unverified". This section
replaces that guess with measured counts. Nothing here is inferred from prose.

**Method.** "Production importers" = `grep -rnE "^\s*(from|import)\s+<module>"`, excluding
`tests/`, `experiments/`, `_archive/`. A module is COUPLED only if a production
file imports it *or* calls its API. A module imported only by `experiments/` bears
evidence but is not on a production path.

**Counts are IMPORT STATEMENTS, not mentions.** A loose `grep -rl <module>` returns
6 files for `henri_scene_binder`; 5 of those are prose (a module docstring at
`henri_curriculum_grid.py:26`, comments at `stage0_seeding_run.py:570,580`, and
three render tools). Only `tools/measure_functional_pipeline.py:24` is an import.
The landing gate re-runs the import-specific pattern, so a mention can never be
counted as a coupling.

| Pillar | Module | Production importers | Production call sites | Verdict |
|---|---|---|---|---|
| 1 | `henri_scene_binder.py` | 1 (`tools/measure_functional_pipeline.py:24`) | `henri_functional_pipeline.py:139,147,149,170` | COUPLED (injected) |
| 3 | `arc_sagnac_veto.py` | 2 (`henri_dual_speed_harness.py`, `production_arc_run.py`) | `production_arc_run.py:2598`, guarded by `HENRI_ARC_SAGNAC_VETO` (default OFF) | COUPLED, default-OFF |
| 3 | `henri_thermo_langevin.py` | 0 | 4 importers, all under `experiments/verification/` | ORPHANED |
| 5 | `zone_c_causal_engram_dag.py` | 0 | 2 importers, both under `experiments/verification/` | ORPHANED (flag default OFF) |
| 5 | `zone_c_attractor_pruner.py` | 0 | 1 importer, `tests/unit/test_architectural_milestones.py` | ORPHANED |

**Finding 1 -- CORRECTED. An earlier revision of this section called the Sagnac
veto ORPHANED. That verdict was WRONG and is retracted here.** `arc_sagnac_veto.py`
IS imported on production paths. `production_arc_run.py:2587` imports it and
`production_arc_run.py:2598` calls it:

```python
from arc_sagnac_veto import apply_advisory_rerank, evaluate_veto
_da, _de, _trig, _st = evaluate_veto(_wave.detach(), _axiom_ref, _world_ref)
```

guarded by `if (HENRI_ARC_SAGNAC_VETO and policy_mode() != "action1" and not psg_engaged)`
where `HENRI_ARC_SAGNAC_VETO = os.environ.get("HENRI_ARC_SAGNAC_VETO", "0") == "1"`
(`production_arc_run.py:278`). The coupling is REAL but DEFAULT-OFF.
`henri_dual_speed_harness.py:64,67` also imports and binds it, and
`henri_functional_pipeline.py:130-131` forwards a `veto_fn` into `AgentialChain`
(`tests/unit/test_functional_pipeline.py:169` asserts a live `sagnac_veto` stage).

The sub-claim that survives is narrower: the file the module's own docstring names
as its consumer, `basal_boundary_engine.py`, calls `evaluate_veto(` **0** times. The
earlier ORPHANED verdict generalised from that one file to the whole codebase --
verification scope too narrow, the same defect class this record polices.

**How the error passed the gate.** The landing gate checked only the sub-claim
(`basal_boundary_engine.py = 0`). A gate that verifies a true sentence can still
pass a false conclusion drawn from it. The corrected gate checks the module-wide
importer count, the specific call site, and the flag default.

**Finding 2 -- the thermodynamic sink is diagnostics-only.** All four importers of
`henri_thermo_langevin` live under `experiments/verification/`
(`thermo_diagnostic.py`, `thermo_gradcheck.py`, `thermo_imprint_sweep.py`,
`thermo_trainer_diag.py`). `arc_sagnac_veto.py` does not import it. The Pillar 3
loop (veto -> Langevin thermalization) does not exist in the code.

**Finding 3 -- the DAG and the pruner are not connected to each other or to
production.** `zone_c_causal_engram_dag.py` contains **0** references to
`attractor_pruner` / `AttractorPruner`. The DAG's `flag_enabled()` is default OFF
(`FLAG_ENV == "1"` required). `zone_c_attractor_pruner.py` is imported only by one
unit test. STEP 4's second half ("connect Zone C DAG to prune previously failed
operator branches") is therefore unsatisfiable by wiring these two: both ends are
orphans, so connecting them would change no production behaviour.

### Consequence for STEP 4, stated precisely

Wiring the DAG to the pruner would connect two modules that no production path
reaches. That produces no behavioural change and is the wrapper-that-only-renames
failure mode rejected in section 2. The prerequisite is a production consumer for
the veto-and-sink loop, not another wrapper.

### What this section does NOT claim

- Not that these modules are defective: each has a tested in-memory API.
- Not that Pillar 3's physics constants are wrong: only that nothing runs them live.
- G2 (`henri_operator_router.py`) was not audited in this pass and is not claimed.

### Pillar scoreboard after this audit

| Pillar | Status |
|---|---|
| 1 Hierarchy | COUPLED via injection; still lacks multiscale Jordan-curve nesting |
| 2 World model | built and MEASURED NEGATIVE at this scale (section 3.3) |
| 3 Sagnac veto + sink | veto COUPLED but default-OFF; sink ORPHANED; the loop is absent (Findings 1 and 2) |
| 4 Hopfield egress | hardened; PROVISIONAL, promotion gate UNMET |
| 5 Causal DAG | components present; not on any production path (Finding 3) |

Two of five pillars remain unmade as loops. Step 4 stays blocked behind Step 3's
0.92 contract, which no measured arm has met.

### Correction log

- 2026-10-01: `arc_sagnac_veto.py` verdict changed ORPHANED -> COUPLED, default-OFF,
  after a MODULE-WIDE importer scan found `production_arc_run.py:2598`. The
  sub-claim about `basal_boundary_engine.py` (0 calls) was and remains true; the
  generalisation from it was not. Gate scope widened from single-file to module-wide.


---

## 7. Pillar 3 loop: built, tested, DEFAULT-OFF, not yet wired (2026-10-01)

Section 6, Finding 2 recorded that `arc_sagnac_veto.py` does not import
`henri_thermo_langevin`, so the Pillar 3 loop did not exist in the code. This
section records the new coupling and, just as precisely, its remaining limit.

**New module:** `henri_sagnac_thermal_loop.py`. Flag `HENRI_SAGNAC_THERMAL_LOOP`,
default `"0"` so the production path is byte-identical when unset.

**Mechanism.** `evaluate_veto(candidate, axiom, world)` -> on a hard veto, the
thermal budget is `kT = kT_base * (1 + excess)` clamped to `kT_max`, where
`excess = max(0, (delta_axiom - tau_veto) / tau_veto)`. Bounded anisotropic creep
runs for `max_steps`, using the SGLD convention `sqrt(2 * gamma * kT * dt)`.
`VETO_UNAVAILABLE` never fires: a failed measurement is not a veto.

**Evidence.** 11 contract tests pass (`tests/unit/test_sagnac_thermal_loop.py`).
An 8-claim probe holds at `experiments/verification/thermal_loop_probe.py`
(`python experiments/verification/thermal_loop_probe.py` -> rc 0).
Reproducibility: identical seed -> bit-identical
creep (`max|diff| = 0.0`). Invariants: `kT` strictly increasing in `delta` and
`> kT_base` when active, bounded by `kT_max` for every tested input, `||theta||`
restored after creep.

### Two defects the RED probe found (both in this session's own code)

1. **Fixture outside the metric's domain.** `_sagnac_similarity` for complex waves
   is `|mean(conj(a) * b)|` -- a MEAN over D components, not a normalised inner
   product. Measured: `ones vs ones -> sim 1.0000` (quiet), but
   `onehot vs onehot -> sim 0.015625 = 1/D -> delta 0.984 -> FIRES`. So `eps_hard
   = 0.35` is meaningful only for near-unit-modulus waves. This is now pinned by
   test T9 rather than hidden, and the probe fixture was corrected.
2. **The helper disagreed with the event it described.** `kT_from_delta` returned
   `kT_base = 1.0` at or below threshold, while the quiescent event reported
   `kT = 0.0` for that same state. Fixed: the helper now returns `0.0` when
   `excess <= 0`, so one state has one thermal budget.

### The limit, stated plainly

**`henri_sagnac_thermal_loop.py` has 0 production importers.** The module exists
and is tested, but no production path constructs it, so the veto -> thermalisation
loop is **still absent from production**. What is now true is that the coupling is
*built and provably correct in isolation*; what remains is a production consumer
that calls it on a vetoed candidate. That is a wiring task against a real module,
not a design question.

### What this section does NOT claim

- Not that the 8.2 line of physics in the source documents is reproduced.
- Not that the loop improves any task metric. No task metric was measured.
- Not that `henri_thermo_langevin` is now coupled: it is still imported only by
  `experiments/verification/`. This module re-implements the noise convention
  rather than importing the sink, because the sink's `LangevinComputer` carries
  trainable couplings and optimiser state that a per-veto creep step must not touch.
