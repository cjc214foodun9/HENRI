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
