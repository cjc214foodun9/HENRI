# Zone A Typed Probe Contract — TypeSafe/Jev Findings Applied to HENRI V2

Status: `IMPLEMENTED — default-OFF. CPU-verified. CUDA verification BLOCKED.`
Date: 2026-09-17
Author: HENRI development arbiter (Hermes)
Repo: `C:\Users\chan\Desktop\HENRI 7B SWARM` @ `carrier/e6-physical-verifier` (`238408c`)
Primary sources: `Project HENRI Architectural Synthesis & Rust Migration Assessment.pdf`
(22 pp.), `Untitled document.pdf` (full TypeSafe docs dump), `docs.typesafe.ai` (live),
`HENRI_V2_dream_vs_reality_audit.md`.
Evidence labels: `OBSERVED` · `DERIVED` · `INFERRED` · `HYPOTHESIS` · `FALSIFIED` · `BLOCKED`.

## 0.1 Implementation status (this revision)

| Item | State |
|---|---|
| `ProbeEnvelope` + `confidence_from_probabilities` + `key⊗value` binding | `IMPLEMENTED` in `arc_egress_contract.py`, default-OFF behind `HENRI_TYPED_PROBE_CONTRACT` |
| Contract tests | `IMPLEMENTED` — `tests/contract/test_typed_probe_contract.py`, **24 passed** |
| Scalar-rotor rejection (A3) | `IMPLEMENTED` + tested; raises `ScalarRotorRejected` |
| Calibration measurement layer | `IMPLEMENTED` — `henri_probe_calibration.py`; skew / sharpness / peak-histogram companions added on THIS project's own finding that ECE alone is gameable by a uniform predictor (the attached document orders only a Brier + ECE receipt) |
| Calibration NUMBER | **MEASURED** — first empirical receipt: `experiments/verification/calibration_eval_observed.json`, 60/60 ARC tasks, schema `henri.calibration-receipt.v1`, `evidence_class: OBSERVED`, `status: OK` |
| Is the head calibrated? | **NO** — functor arm acc 0.7833 vs mean peak 0.3011, ECE **0.4822**, skew **+0.4822** (positive = UNDERCONFIDENT), `is_well_calibrated: False` |
| Uniform-predictor trap, live | the `random` arm scores ECE 0.0487 (LOWER) with `brier_skill_score` −0.0001 (negative) — exactly why the joint gate is used instead of ECE alone |
| Encoder-basis receipt (approval 3) | `IMPLEMENTED` — `phase_map_basis_observed.json` + doc, registered as seal pair 7 |
| Phantom-receipt defect (§9.0) | **CLOSED** — receipt generated; docstring counts corrected 16/14 → 27/16 |
| CUDA verification | `BLOCKED` — Vast SSH refused, instance EXITED, credit 0 |
| VLA SOTA on AAII v4.3 | `BLOCKED` — no harness, no compute; see §10 |

**The word "calibrated" is now `FALSIFIED` for this head, not merely unverified.** The
number exists and it is bad: the readout is *underconfident* (it is more accurate than it
says it is), and the joint gate `ECE <= 0.05 AND brier_skill_score > 0` rejects it.
A falsified label is a stronger and more useful result than an open `HYPOTHESIS`, and it
was produced by measurement, not assertion.

The receipt's `evidence_class` is `OBSERVED` at the receipt level: per-task wave cosines
are measured on the live encoder, and all 60 input rows are retained in `per_task` so the
aggregates are independently recomputable.

---


## 0. Decision

Adopt the TypeSafe core philosophy **as a boundary contract, not as a model**:

> Give up open-ended string generation at the HENRI/world boundary. Make every
> crossing a typed, calibrated decision whose uncertainty is a first-class value
> the code can branch on.

This is **correct**, and it is a **pruning and wiring task**, not a new subsystem.
The repository already holds most of the required machinery. Two specific claims in
the synthesis PDF that justify this decision are `FALSIFIED` against live code
(section 2). One mechanism the PDF proposes as the fix is a **no-op** (section 3).

---

## 1. What TypeSafe actually contributes (and what it does not)

| TypeSafe mechanism | Evidence in TypeSafe docs | HENRI contract it settles |
|---|---|---|
| Typed question against a `state`; no text out, no parsing back | `OBSERVED` (`/introduction`, `/concepts/system-one`) | Zone A egress must emit a struct, never a string. |
| Three primitives: `Choice`, `Score`, `Noul` | `OBSERVED` (`/primitives`) | The complete answer alphabet for a boundary crossing. |
| `probabilities` + derived `confidence` on every Choice/Score | `OBSERVED` (`/confidence`) | Uncertainty is a second decision axis, not a diagnostic. |
| Questions evaluated **in isolation against one frozen state**, in parallel | `OBSERVED` (`/primitives/choice`, `/cookbooks/classification_using_confidence`) | Probes must read a **pinned snapshot**, or parallel results are mutually incoherent. |
| Decompose multi-factor judgments; recombine in code | `OBSERVED` (`/introduction`) | No multi-factor mega-head. Weight changes are code edits, not retraining. |
| Calibration is **earned against outcomes** (RLCD), not declared | `OBSERVED` (`/introduction/machine-learning-primer`) | Deploying Jev grants structure, **not** calibration. |

### 1.1 What HENRI does NOT inherit

`HENRI cannot inherit calibration from Jev.` Jev's calibration comes from RLCD training
against outcomes (`OBSERVED`, ML primer). A HENRI head that copies the *interface* gains
structure, composability, and observability. It does **not** gain the property that makes
the interface useful.

Measured absence (`OBSERVED`): a repository-wide search for
`brier|expected_calibration|ece_|reliability_diagram|calibration_error` over
`HENRI V2/**` returns **0 matches**. HENRI has never computed a probability-calibration
statistic.

The nearest thing named "calibration" is `henri_calibrated_action_head.py`, whose
`E_cal <= 0.05` gate is a **held-out MSE threshold** (`OBSERVED`, lines 1–120). An MSE
fit gate is not probability calibration. It cannot tell you whether a 0.8-confidence
Choice is right 80% of the time.

**Therefore:** "calibrated epistemic probe" is currently `HYPOTHESIS`. The word
`calibrated` is `unverified` until a reliability measurement exists.

---

## 2. Corrections to the synthesis PDF (verified against live code)

The PDF's conclusions are broadly right. Two of its **citations** are wrong, and one of
its **fixes** is a regression. Correcting them matters, because both errors would send
implementation the wrong way.

### 2.1 `FALSIFIED` — the `torch.abs()` root cause is a phantom

The PDF states (pp. 13, 21) that Defect A2 was caused by taking the Euclidean magnitude
`torch.abs(psi_wave)` inside `henri_decoder.py`, "erasing the phase angle".

Measured (`OBSERVED`): `torch.abs` does **not** occur in `henri_decoder.py`.
`grep -rn "torch\.abs" HENRI V2/*.py` returns zero hits in that file. The decoder's
forward pass is `down_proj → layer_norm → GELU → lm_head` on `wave_state.real`
(lines 110–113). There is no magnitude step before projection.

Real `torch.abs` sites (`OBSERVED`):

| Site | Context | Verdict |
|---|---|---|
| `arc_task_functor.py:281,282` (`_f7_held_cos`, `_f7_identity_cos`) | Absolute value applied to a cosine, then asserted against a pinned constant | This is the exact code path behind the **f6/f7 stale-pin failures** documented in the audit. |
| `arc_sagnac_veto.py:50` | `torch.abs(torch.mean(a.conj()*b))` — complex unit-modulus branch | Correct. This is the canonical qFHRR metric, not a defect. |

**Consequence:** do not dispatch a "fix `henri_decoder.py`'s `torch.abs`" task. It has no
target. The A2 *symptom* (`top1_token_unique = 1`, mutual-information collapse) is real
and `OBSERVED`; the cited *mechanism* is not. Section 3 supplies the mechanism that does
survive measurement.

### 2.2 `CONDITIONAL` — the Torus Adjoint is invertible only on a non-default basis

The PDF asserts (p. 12) the representation substrate is "MATHEMATICALLY CLOSED &
INVERTIBLE — 8/8 exact grid decode". The 8/8 receipt is real. The claim is stated
unconditionally. It is not unconditional.

Measured (`OBSERVED`):
- `henri_vision_encoder.py:82` — in the **default** basis, `spatial_phases_y = spatial_phases_x`.
  The position carrier is therefore `exp(i(x+y)w)`, which is rank-deficient. Single-pixel
  grids at `(1,2)` and `(2,1)` produce **identical** waves.
- `arc_phase_map.py:38-40` registers exactly this as
  `BLOCKED_PHASE_MAP_NONINVERTIBLE`.
- `henri_vision_encoder.py:67-80` — a `spatial_basis_kind` of `incommensurate` or `random`
  replaces the `y` ramp and restores separability. Phase 7.3 G1 is recorded `ACCEPTED`.

**Consequence:** "the wave core is invertible" is true **only** for
`spatial_basis_kind ∈ {incommensurate, random}`. The default path is `BLOCKED` for 2D
localization. Do not cite 8/8 as evidence about the default path. Promote the basis kind
behind a flag with a paired receipt, or the claim stays conditional.

### 2.3 `STALE` — the `carrier/aaii-v43 → main` merge is already partly done

The PDF's Step 1 says to merge `carrier/aaii-v43` into `main` to "eliminate stale
baseline drifts".

Measured (`OBSERVED`, 2026-09-17):
- `main` = `090fe87` (2026-09-16)
- `carrier/aaii-v43` = `16d573c` (2026-09-17)
- `git merge-base --is-ancestor carrier/aaii-v43 main` → **false**; `git rev-list --count
  carrier/aaii-v43 ^main` → **4**

A fast-forward promotion already occurred. Four new commits are pending, the newest being
`feat(resonator): tripartite VSA resonator — instrument VALIDATED (28/28, mutation gate
4/4), VOID on real ARC`. Note the honest negative: the resonator is **VOID on real ARC**.

**Consequence:** the pending action is a 4-commit promotion with a fail-closed gate, not
a first-time merge. Do not describe `main` as stale.

### 2.4 `REGRESSION` — the PDF's probe codebook reintroduces the failed head

The PDF proposes a 64-probe manifest over `S^{feat_dim-1}` with random prototypes
(`torch.randn(seed=42)`), plus a 32,000-entry token vocabulary in the Rust port
(`unbinder.rs`, `vocab_size`).

The repository already settled this (`OBSERVED`, `arc_egress_contract.py` docstring,
contracts 3 and 4):

> "The 32k code-token vocabulary is NOT an action vocabulary. The action-legal vocabulary
> occupies the FIRST N logit positions… Positions ≥ N are code tokens and are never
> interpreted as actions."

The PDF re-proposes a 32k-entry probe codebook. That is the A2 failure surface again, and
it contradicts a written contract in the live tree. Reject it.

### 2.5 `FABRICATED` — the proposed probe body is a mock loop

`TypedEgressProber` as printed in the PDF (pp. 7–11) is not a prober. It is a mock.

| Construct | Measured status |
|---|---|
| `expected_information_gain_nats = 1.5 + (i % 5) * 0.3` | Arithmetic literal. Never measured. No environment, no entropy estimate. |
| `action_primitive = "TEST_BOUNDARY" if i % 2 == 0 else "INSPECT_ATTRACTOR"` | Fabricated affordance names. No environment exposes these. |
| `codebook = torch.randn(64, feat_dim, seed=42)` | Random attractors. Carries no semantics. |
| `sagnac_stress = 0.0215` | Hardcoded literal, independent of `psi`. |
| `is_safe_to_execute = sagnac_stress <= 0.0431` | Tautology. `0.0215 <= 0.0431` is always true, so `VETOED_BY_SAGNAC` is unreachable. |
| `0.0431` | Appears nowhere in HENRI. The repo's own hard threshold is `epsilon_hard = 0.35` (`arc_sagnac_veto.py:38`). |

This is exactly the pattern the arbiter filter rejects: a placeholder payload standing in
for a real gateway, a diagnostic channel used as an objective, and a veto that can never
fire. The PDF's *prose* is sound; its *code* must not be copied.

---

## 3. The measured mechanism: why the proposed feedback is a no-op

This is the load-bearing section. The PDF's step 4–5 (pp. 10) is:

```python
phase_delta = (obs_feedback * 2.0 * math.pi) % (2.0 * math.pi)
rotor       = torch.complex(torch.cos(phase_delta), torch.sin(phase_delta))
psi_updated = F.normalize(psi_current * rotor, p=2.0, dim=-1)
```

**Measured (`OBSERVED`, isolated CPU probe, reduced `D = 4096`; the claim is
scale-invariant).** Reference implementation: `falsify_prober_v3.py`, `t2_corrected.py`.

| Quantity | Scalar typed answer | Per-dimension (per-block) answer |
|---|---|---|
| `‖psi‖₂` before | 64.000000 | 64.000000 |
| Normalized overlap `\|⟨ψ',ψ⟩\| / (‖ψ'‖‖ψ‖)` | **0.999999940** | **0.021205** |
| Raw `‖ψ' − ψ‖₂` | **64.3175** | large |
| HENRI Sagnac readout on the pair | S = 0.015625 → **δ = 0.984375** | δ ≈ 0.9997 |

**Finding.** A typed answer that arrives as a **scalar** produces a **scalar** rotor. A
scalar rotor is `e^{iθ}`, which is a **global phase** — a `U(1)` gauge transformation. It
is invisible to every phase-invariant readout. The raw vector moves by `‖ψ'−ψ‖ ≈ 64`, so
naive instrumentation reports a large update while the **state is bit-identical in every
physically meaningful sense**.

Two consequences, both severe:

1. **A scalar-rotor feedback loop is a mock loop.** It performs "incessant learning" that
   cannot change any decision, then reports a large `‖Δψ‖`. Per the arbiter rules, this is
   a diagnostic channel wearing an objective's clothes.
2. **It manufactures false vetoes.** A `‖ψ‖₂ = √D` vector scored against a *normalized*
   copy gives `S ≈ 1/√D`, hence `δ ≈ 1`. This is the exact defect the repository already
   documents **twice**, independently:
   - `arc_sagnac_veto.py:8-15` — "delta ~ 0.99998 at D=65,536… vetoes every valid candidate."
   - `basal_boundary_engine.py:10-13` (`D-SAGNAC`) — "divides the real inner product of two
     UNIT-NORM waves by D… `1 - inner/D` is ~0.99998."

   Two modules in the live tree already record this trap. The PDF's `sagnac_stress` literal
   (0.0215) conceals it; a computed stress would expose it immediately. This is the
   strongest single reason to reject the PDF's code body while accepting its philosophy.

### 3.1 The fix that does work

A typed answer must be **bound into all `D` dimensions** as a VSA key⊗value pair before it
enters the wave core. Measured recovery over `K = 8` candidate answers, 16 trials:

| Bridge | Recovery | Note |
|---|---|---|
| Scalar (gauge) rotor | 2/16 = 0.125 | Exactly chance. The answer is not in the state. |
| Key only, value discarded | 4/16 = 0.250 | Slightly above chance; the value is lost. |
| **`key ⊗ value` bound into `D` dims** | **16/16 = 1.000** | The typed answer is decodable from the state. |

Chance = 0.125. The bound form recovers the answer on every trial; the scalar form cannot
recover it at all. This is the concrete, falsifiable content of "feed continuous phase
rotations back into the unitary core": the rotation must be **per-dimension**, and the
answer must be **bound**, not broadcast.

**This is consistent with the repo's own invariant** (`memory`, `arc_egress_contract.py`):
phasor binding is order-sensitive, and the codebook must be tokenizer-derived. A random
64-prototype codebook (PDF §2.5) cannot be inverted because nothing binds it to meaning.

---

## 4. The contract

### 4.1 Proposed `ProbeEnvelope`

One struct. Every Zone A crossing emits it. No `str` reaches a machine consumer.

```
ProbeEnvelope:
  probe_id            : int                 # stable id, registered in a manifest
  question_type       : {"CHOICE","SCORE","NOUL"}
  state_snapshot_id   : str                 # content hash of the PINNED wave state
  option_ids          : [int]               # CHOICE only; bounded, non-empty
  probabilities       : [float]             # sums to 1; length == |options| or |levels|
  answer              : int | float         # argmax option / score value / p(True)
  confidence          : float in [0,1]      # DERIVED from probabilities, defined once
  wave_binding        : key ⊗ value         # [num_blocks, 8] — the ONLY feedback path
  status              : {"OK","ABSTAIN_LOW_CONF","ABSTAIN_NO_ORDER","ABSTAIN_INVALID"}
  provenance          : {run_id, arm_id, commit_sha, source_hash}
```

Rules (each fails closed):

1. `status != "OK"` ⇒ the answer is **not** consumed. Abstention is a valid terminal result.
2. `confidence` is computed from `probabilities` by **one named function**, so it can be
   recalibrated in one place. Do not scatter thresholds.
3. `wave_binding` is the sole feedback channel. Reject any code path that applies a scalar
   rotor (section 3).
4. `state_snapshot_id` is mandatory. Parallel probes on a rotating state are incoherent
   without it (TypeSafe's isolation property).

### 4.2 Disposition of existing egress paths

The repo already holds ~10 egress modules. Reuse them; do not add a parallel stack.

| Module | Role today | Disposition |
|---|---|---|
| `arc_egress_contract.py` | Typed action egress: action-legal logits, `probs`, `top3`, `entropy_bits`, fail-closed `EgressFailClosedError`, `NoDemonstrationsError` | **KEEP — this is the base.** It is already the contract shape. Add `ProbeEnvelope` + confidence. |
| `henri_hopfield_egress.py` | Modern Hopfield snap, β=8.0, returns `REJECTED` (never a fabricated token), `noise_floor_sigma = eps/√D` | **KEEP.** The `ε/√D` normalization is the correct dimension-aware form. Reuse its β and its dimension-aware floor. |
| `g5_semantic_egress.py`, `g5_separable_codec.py`, `g7_highorder_codec.py` | Decode with explicit `ABSTAIN_*` statuses; "never fabricates partial text" | **KEEP.** Abstention discipline is the `status` vocabulary for §4.1. |
| `henri_calibrated_action_head.py` | Stiefel-ridge head, `E_cal` MSE gate, default-OFF, `BLOCKED_NO_ACTION_TRAJECTORIES` | **KEEP, RENAME THE GATE.** `E_cal` is a fit gate, not calibration. Do not cite it as calibration evidence. |
| `henri_decoder.py` | 2-layer `lm_head` to 32,000 logits | **DEMOTE.** Stop calling it a generator. It is an uncalibrated readout head. It is `RETRAIN_REQUIRED` (A2). |
| `wave_ast_decoder.py`, `henri_ast_grammar_mask.py` | Grammar-masked AST for HumanEval/MBPP | **KEEP.** Structured output is already typed. Do not conflate with free-text generation. |
| `efe_planner.py` | EFE = pragmatic − epistemic per candidate | **KEEP — this is the probe *selector*.** It already computes epistemic value. |
| `zone_c_epistemic_axiom_harness.py` | Epistemic axiom harness | **KEEP.** Inspect before adding a new probe registry. |

**Do not create `henri_active_prober.py` as the PDF specifies.** That would be a duplicate
wrapper over `arc_egress_contract.py` + `henri_hopfield_egress.py` + `efe_planner.py`,
which the arbiter rules classify as a wrapper that renames existing operations and calls
it new capability.

---

## 5. The bounded change and its kill experiment

**One change.** Add `ProbeEnvelope` + a single named `confidence` function +
`wave_binding` construction to `arc_egress_contract.py`, behind a named default-OFF flag.

```
HENRI_TYPED_PROBE_CONTRACT=1     # default unset -> existing path byte-identical
```

**Pre-registered acceptance** (must pass on the Vast CUDA target, not CPU):

- **A1** Default path byte-identical when the flag is unset (existing action-legal logits,
  `top3`, `entropy_bits` unchanged).
- **A2** `status == "OK"` ⇒ `sum(probabilities) == 1 ± 1e-6` and `answer == argmax`.
- **A3** Scalar-rotor path is **unreachable**: an AST-lint or runtime assert rejects any
  feedback that applies a shape-`[1]` rotor to a `D`-dim wave.
- **A4** Bound recovery: `key ⊗ value` recovers the probe answer on ≥ 15/16 trials at
  `D = 65536`, against a chance baseline of `1/K`. (CPU proxy measured 16/16 at `D=4096`.)

**Pre-registered rejection** (any one kills the change):

- **R1** `confidence` is not monotone in `probabilities` peak height.
- **R2** `wave_binding` fails to distinguish two different answers (normalized overlap
  `> 0.9` between the two bound states).
- **R3** Any probe result is consumed while `status != "OK"`.
- **R4** The default path changes when the flag is unset.

### 5.1 R2 kill experiment — EXECUTED (2026-09-17)

Ran at **production scale `D = 65,536`**, `K = 8` answers, 28 pairs.
Script: `falsify_prober` family, `/r2_kill.py` (CPU, no HENRI import).

| Quantity | Value |
|---|---|
| Pairwise normalized overlap, **max** | **7.349e-03** |
| mean | 3.268e-03 |
| min | 4.634e-04 |
| chance level `1/√D` | 3.906e-03 |
| **R2 verdict** | **DOES NOT FIRE** (max 7.35e-3 ≪ 0.9) |

`OBSERVED`. Bound states for different answers are effectively orthogonal, so the
feedback channel does carry answer identity. **The concept survives its cheapest kill.**

**A4 bound recovery at production scale:** `16/16 = 1.000` (threshold ≥ 15/16). Gauge
scalar-rotor control: `1/16 = 0.062` (below chance 0.125).

### 5.2 A4 caveat — circular validation risk (must not be suppressed)

`R2/A4 as I ran them are NOT sufficient evidence for the production claim.` The readout
basis in the A4 decode **is the same basis that encoded the answer** (`key`, `vals`). Exact
recovery therefore follows partly *by construction*. This is the circular-validation
pattern the arbiter filter rejects: scoring the system with the signal that built it.

What the executed probes DO establish (`OBSERVED`):
1. A scalar typed answer is a pure gauge rotation — state content unchanged (overlap
   0.999999940). **This is airtight and is the decisive negative.**
2. A per-dimension bound rotation changes the state, and distinct answers give mutually
   orthogonal states (max overlap 7.3e-3). **Also solid.**
3. An *identity* codebook recovers its own binding. **This is the part that proves little.**

What remains `HYPOTHESIS` and is required before any production claim:
> A **learned or tokenizer-derived** codebook (not a random `randn` codebook, not the
> construction basis) recovers the correct answer from the wave on **held-out** probes,
> above chance, with the codebook frozen before evaluation.

Acceptance for that stronger test: **≥ 15/16 on a held-out probe set with a frozen
non-identity codebook**, and the random-prototype codebook must **fail** the same test.
The second half matters most: if a random codebook also scores 16/16, the test is
measuring the readout, not the representation.

**Cheapest kill experiment.** R2 alone is now closed. The next-cheapest kill is the frozen-codebook
held-out test above. If a *random* codebook scores as well as a structured one, the probe
representation carries no meaning and the change should be abandoned.

**Not in scope for this change:** Rust. Do not scaffold `henri-substrate` while the
operator algebra is mutating. Rust becomes the right home for this contract *after* the
contract is stable, and it is then a genuine gift: a tagged enum
`Probe { Choice(..), Score(..), Noul(..) }` makes "no `String` on the machine path"
a compile-time guarantee rather than a lint.

### 5.3 Placement — where the contract lives

`OBSERVED`: the Zone A boundary object **already exists**.

`basal_boundary_engine.py` (834 L) documents itself as "Zone A dynamic Markov blanket +
Zone B basal oscillator syncytium" and defines `DynamicMarkovBlanket` (line 674):

> "The blanket does not compute a solution. It decides WHERE (tile), WHEN (slot) and HOW
> LONG (relaxation steps) the basal channels observe and update against the external task."

That is a probe **scheduler** — the aperture, gate, and timing for crossing into the world.
It already carries `sagnac_epsilon` and `r_gate` as constructor parameters.

**Therefore the probe contract is not a new subsystem.** It is:
1. `ProbeEnvelope` + `confidence` in `arc_egress_contract.py` (**what** crosses), and
2. a probe-selection method on `DynamicMarkovBlanket` / `efe_planner.py` (**when** it crosses).

Do not introduce a third boundary object. Two schedulers with different timing semantics is
the duplicate-wrapper failure.

### 5.4 Premature discretization — a trap to avoid

TypeSafe returns **distributions**, not labels. Keep the wave state continuous internally;
quantize only at the egress boundary. In particular: do **not** run a
`cegis_grid_snap.py`-style hard snap on a candidate before its confidence is recorded. A
snap applied early destroys the confidence signal, and the whole point of the contract is
that confidence survives to the consumer.

### 5.5 Terminology

"Choice / Score / Noul" is TypeSafe's brand vocabulary. Use **"choice / graded / truth
probe"** in HENRI code and docs to avoid cargo-culting a vendor's naming into the
architecture. Keep the *semantics*; drop the *labels*.

---

## 6. What must not be lost

1. **Do not delete generation — relocate it.** `henri_semantic_backbone.py` and
   `henri_backbone_adapter.py` still formulate probes from goals and render results for
   humans. Generation belongs on the *observation* side. Never on the *decision* side.
2. **"Incessant" needs a stopping rule.** `efe_planner.py` ranks by expected free energy.
   Add a floor: stop when expected information gain per probe falls below a threshold, or
   the loop degenerates into unbounded query churn. The user's word "incessantly" must be
   bounded by a budget, not by enthusiasm.
3. **Ground outcomes externally.** The dominant failure mode of a probe-and-feed-back loop
   is self-confirmation: the core consumes only its own interpretations. Outcomes must
   come from external verifiers — sandbox execution (`mbpp_secure_executor.py`,
   `exegis`-style CEGIS), ARC ground truth, benchmark harnesses. This is consistent with
   the standing rule that internal coherence and external outcome are separate evidence
   streams.
4. **NFL boundary holds.** Do not put factual world knowledge into un-parameterized wave
   oscillators. It belongs in the frozen adapter. The probe contract does not change this.

---

## 7. Evidence status

| # | Claim | Class |
|---|---|---|
| 1 | TypeSafe/Jev provides a typed, calibrated decision interface | `OBSERVED` (docs) |
| 2 | HENRI should not be an NLG at its transduction boundary | `OBSERVED` (A2 `top1_token_unique = 1`) |
| 3 | Scalar typed answer ⇒ pure gauge rotation ⇒ state unchanged | **`OBSERVED`** (overlap 0.999999940) |
| 4 | Typed answer bound into `D` dims ⇒ decodable | **`OBSERVED`** (16/16 vs chance 0.125) |
| 5 | `torch.abs` in `henri_decoder.py` caused A2 | **`FALSIFIED`** (no such call exists) |
| 6 | Torus adjoint always invertible | **`CONDITIONAL`** (fails on default basis) |
| 7 | `carrier/aaii-v43 → main` merge pending | **`STALE`** (FF done; 4 commits remain) |
| 8 | PDF's `TypedEgressProber` is a working design | **`FALSIFIED`** (literals, tautological veto) |
| 9 | HENRI probes are "calibrated" | **`HYPOTHESIS`** (0 calibration metrics exist) |
| 10 | `arc_egress_contract.py` is a suitable base for the contract | `OBSERVED` |
| 11 | Verification on CUDA | `BLOCKED` (Vast instance EXITED, credit 0) |

---

## 8. Uncertainty

- All numeric probes ran on **CPU** at `D = 4096`, `feat_dim = 256`. The claims are
  structural and scale-invariant, but they are **not** CUDA-verified at `D = 65,536`.
- I did not run `production_arc_run.py` (needs `arc_agi`, `arcengine`, CUDA).
- Vast instance `50797414` is EXITED with a negative balance. No remote verification was
  possible. Production Zone C (`:10100`) is unreachable.
- `Untitled document.pdf` is a full TypeSafe documentation dump, not a HENRI document. Its
  content matches `docs.typesafe.ai` and adds no independent evidence.
- The PDF's `0.0431` threshold has unknown provenance. Its origin was not determined.

---

## 9. Next actions (numbered, bounded)

0. **`GOVERNANCE DEFECT — phantom receipt (found while verifying this brief).`**
   `henri_discrete_egress_flag.py:24` cites its evidence as
   `receipt e6_d2_ast_sites.json sha 6e4356bfda9bc12c`, for the claim "16 live construction
   sites in 14 files (AST-measured)".

   Measured (`OBSERVED`, 2026-09-17):
   - `find . -name "e6_d2_ast_sites*"` over the whole repo → **no match**.
   - `git log --all -- "*e6_d2_ast_sites*"` → **no commit ever touched it**.
   - `git check-ignore` → **rc=1**, so it is not a hidden gitignored overlay.

   The **receipt does not exist**. The citing file is a live governance flag whose *guard
   behavior* is real and verified (unset→inert, `"1"`→raise; 4 constructor guards reachable
   in `henri_decoder.py` and `henri_ast_grammar_mask.py`). So this is **not** a broken
   control. It is a **citation to evidence that was never committed** — the exact
   "receipt-only, doc-diverges-from-receipt" class the seal gate exists to catch.

   **Action:** either commit the AST receipt, or downgrade the docstring to state the count
   as unverified. Do not leave a governance flag asserting a measured number with a dangling
   hash. This is the same failure class as the f6/f7 stale pins (audit §9.1): the
   falsification machinery is sound, the *pin* was not maintained.

1. **Approve or reject** the `ProbeEnvelope` addition to `arc_egress_contract.py`
   (default-OFF, named flag). This is the only load-bearing change proposed.
2. **Run the R2 kill experiment** before anything else. One CPU script. If it fails, stop.
3. **Promote the 4 pending `carrier/aaii-v43` commits** to `main` with a fail-closed gate,
   after the f6/f7 stale pins are advanced in the same commit (audit §9.1).
4. **Decide the calibration path.** Three options, needs an explicit decision:
   (a) use Jev as an external oracle to generate calibration data for HENRI heads;
   (b) train a local head under a proper scoring rule with an ECE gate;
   (c) hybrid. Do not ship the word "calibrated" until one of these produces a number.
5. **Promote `spatial_basis_kind`** behind a flag with a paired receipt, or keep the
   invertibility claim `CONDITIONAL`.
6. **Do not** scaffold Rust, **do not** create `henri_active_prober.py`, **do not** copy
   the PDF's code bodies.
