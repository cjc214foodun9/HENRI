# HENRI End-to-End Pipeline Audit — Plain-Language Edition

Date: 2026-09-18. Audit-window base commit: `0fc3df3` (= `origin/main` at audit start).
Evidence labels used throughout: OBSERVED / DERIVED / INFERRED / HYPOTHESIS / FALSIFIED / BLOCKED.
Every number below comes from a live code read or a committed receipt from this session.
Nothing is copied from a design document without a receipt behind it.

## 0. One paragraph

HENRI turns a picture of a grid into a very long list of numbers called a *wave* — 65,536
coordinates arranged as 8,192 rows of 8, where each row is forced to have length exactly 1.
Geometrically that is a direction on the surface of a sphere in 65,536 dimensions, so
"thinking" is rotation and "similarity" is angle. It learns "if I see this, do that" by
comparing the waves of demonstration grids, picks actions by guessing which one makes the
world look most like a solution, rejects impossible guesses with an interference test it
calls the Sagnac veto, and finally snaps a continuous answer onto a discrete choice with an
associative memory. All of that is real, tested, committed code. What does **not** exist is
any mapping from waves to *word meanings*, words, or facts. The system is a working set of
reflexes with no language cortex — which is why the live external score is 0.0%.

## 1. The data path, stage by stage

### 1.1 Ingress — grid to wave (`henri_vision_encoder.py`)
OBSERVED. `HENRIVisionEncoder.encode_spatial_grid` maps a 2D integer grid to a real
`[num_blocks, 8]` wave (8192x8 at production). Each cell's colour value multiplies into a
position wave built from **incommensurate** spatial frequencies (one axis 1.0, the other
sqrt(2)-1 — an irrational ratio, so the two never re-synchronise). Consequence: every (x, y)
coordinate gets a unique interference signature, and the encoding is invertible — measured
16/16 exact coordinate recovery with distinct-position cosine -0.001853 (Seal Pair 7).
Background masking (`bg_mask=True`) removes the constant colour-0 "tablecloth" carrier so
actual shapes dominate the wave. This is interferometry used as an address scheme.

### 1.2 Dynamics — "what happens next" (`efe_planner.py`)
OBSERVED. `LowRankCoupledTransition` is a Koopman/EDMD operator. It lifts state+action into
a larger feature space through complex binding, then fits a **linear** map there. That is the
Koopman trick: nonlinear dynamics become linear if you choose the right lift, and HENRI uses
its own binding operation as the lift. Storage stays factored (`V @ W^T`, rank-bounded) —
a dense 65,536x65,536 operator would be 34 GiB. Training: `train_transition_batch` (least
squares in the sample-dual space, never forming the primal Gram) and `train_transition_step`
(one SGLD update). After every update the factors are retracted back onto the Stiefel
manifold (orthonormal columns) with a Cholesky retraction. Measured engagement this session:
17 real ARC pairs, batch loss 0.984583 — it learns, from very few examples.

### 1.3 Planner — "which action is best" (`EFEPlanner.select_action`)
OBSERVED, and this is the live action policy. Each candidate action yields a predicted next
wave, scored by Expected Free Energy: epistemic value (how much will I learn) plus pragmatic
value (how near is my goal) minus a penalty for violating learned invariants. A calibrated
switch (T4) reads the spread of EFE across candidates: when the spread is large the model
does not trust its own dynamics and takes the most informative action instead of the greedy
one. This is Active Inference used as a control policy.

### 1.4 The Sagnac veto — the physics referee
OBSERVED as a mechanism. `delta_sagnac = 1 - cosine(prediction, constraint_wave)`, a bounded
residual; above threshold the candidate is rejected outright (`Q -> -inf` in the MCTS layer,
`tau_veto` = 0.35 there; 0.0431 in the VLA engine; epsilon in the egress contract). The
constraint waves are 11 frozen unit-norm vectors in Zone C encoding elementary invariants
(translation, rotation, parity, gravity, counting, containment, conservation) — Spelke-style
core knowledge, stored in Postgres/TimescaleDB with pgvector HNSW search under 2.5 ms. The
name is borrowed from an optical interferometer; the implementation is a bounded similarity
gate with fail-closed semantics.

### 1.5 Egress — wave to decision
OBSERVED, and this is the honest bottleneck. Three paths:
- **Hopfield lexical snap.** A modern associative memory: the nearest stored pattern wins,
  beta=8 making it nearly a hard argmax. The sealed, checkpoint-free version retrieves
  **perfectly** — 32000/32000 identity round-trip over the sealed 32k manifest, random-codebook
  control 0/32000. But its **readout cannot decide**: measured margin 0.017557 on encoder
  waves and 0.018518 on EFE-planned waves, against a same-scale random control of 0.024404
  (D=512), with normalized entropy 0.9906 / 0.9907 vs random 0.9891. It is very close to
  uniform.
- **Typed probe contract** (`arc_egress_contract.py`): fail-closed decoding with calibrated
  temperature, top-k margins, entropy, and a decisiveness guard. Machinery is built and
  tested; the readout it gates is the weak part above. Flag `HENRI_SEALED_ACTION_EGRESS`
  is default-OFF, and stays OFF because the measurement does not justify it.
- **CEGIS/REPL**: candidate Python programs are executed in a sandbox and admitted on
  evidence. This is the most complete egress path, bounded by the text codec (§3).

### 1.6 Memory — Zone C (`zone_c_*`)
OBSERVED. A Postgres/TimescaleDB store holding *waves*, not text: boundary axioms, engrams,
and error waves transduced from crashed programs. Hypertables, continuous aggregates over
Sagnac delta, an "apoptosis" job that deletes engrams below a strength threshold, HNSW cosine
search. It is a constraint bank the planner bounds off against — never a training corpus
(governance: Zone C holds frozen engrammatic priors, not benchmark data).

### 1.7 Telemetry — the receipts
OBSERVED. Every step emits a hash-chained JSONL event (state kind, tensor shape, Sagnac
delta, coherence, transition loss, selected action, external outcome) with SHA-256 parent
links. That is why this audit can cite numbers at all.

## 2. Hardware: what exists, and what is only a target

This section is deliberately blunt, because the design documents describe hardware that is
not built.

- **Present (OBSERVED):** commodity x86 CPU running PyTorch in float32/complex64, plus
  Postgres/TimescaleDB on localhost for Zone C. That is the entire physical substrate in use.
- **Absent (BLOCKED, 11 claims / 0 measured):** thin-film BaTiO3 or LiNbO3 photonic
  integrated circuits, Mach-Zehnder meshes, TiN micro-heaters, CXL 3.0 zero-copy DMA, Rust
  dispatch. Numbers such as "< 0.1 ns optical transit" and "< 6.67 nJ/step" are **targets
  with no measurement**, and the energy figure is `None` by construction because no
  instrument here can measure joules per step.
- **The key honest framing:** HENRI does not *run on* physics. It **simulates** an
  interference computer on an ordinary von Neumann machine. Unitarity, interference, and
  phase synchronisation are arithmetic performed on tensors — they are the *algorithm*, not
  a property of the hardware. Any claim of substrate-level advantage is `HYPOTHESIS`.
- **GPU (BLOCKED):** the production-dimension path needs an NVIDIA GPU. The instance is
  terminated (credit 0), and the production planner additionally requires a checkpoint
  overlay that is absent from this checkout.

## 3. The mathematics, in plain words

- **Hypersphere state.** Each thought is a unit-length direction; thinking is rotation,
  comparison is an angle. Norm preservation (Stiefel retraction, Cholesky) keeps every state
  on that sphere to ~1e-6.
- **Binding by multiplication.** Concepts combine by element-wise complex multiplication,
  which adds phases. That makes binding order-sensitive and approximately invertible
  (unbinding = phase subtraction instead of division). This is Fourier/holographic algebra,
  not a metaphor.
- **Incommensurate carriers.** Two axes advancing at an irrational frequency ratio never
  repeat, so position addresses stay unique — the same principle as coprime gear teeth.
- **Koopman lift.** Learn the change as a matrix in a lifted space rather than as a
  nonlinear map in the original space. Effective rank is `min(requested, N)`.
- **Langevin/SGLD.** Gradient descent plus calibrated thermal noise `sqrt(2*T*dt)` with
  unit-normalised noise, then a manifold retraction. The noise helps escape shallow minima;
  the retraction keeps the geometry legal.
- **Expected Free Energy.** A single sorting key combining "how surprised will I be" and
  "how much do I want this". Active inference as a policy.
- **Sagnac veto.** A bounded cosine residual against frozen invariant waves, used as a
  fail-closed filter. Physical inspiration, mathematical implementation.
- **Hopfield snap.** An energy landscape over stored patterns; the nearest valley wins.
- **Boundary discipline (measured this session):** a *scalar* phase rotation is a gauge
  no-op for selecting among candidates (overlap 0.99999994) — a scalar delta must be spread
  across dimensions as a bounded ramp and recovered against a reference wave to carry
  information at all.

## 4. Measured vs assumed

**Measured and holding:** encoder invertibility 16/16; Hopfield retrieval 32000/32000 vs
random 0/32000; phase sensitivity of the readout (max |delta logit| 16.0 under a pi rotation);
causal consumption of waves by the decoder (22/24 token-byte changes under a seeded rotation,
byte-exact restore); transition learner engages (loss 0.984583 on 17 real pairs); delta
round-trip error 4.29e-06; full suite 2106 passed / 21 skipped / 0 failed; hash-chained
telemetry.

**Measured and weak:** readout decisiveness is below a same-scale random control on every
distribution tested (0.017557 encoder, 0.018518 EFE-planned, 0.024404 random), entropy ~0.99
of 1.0; the demo-path success criterion was answer-coupled and its banner is now removed;
sparse-feedback semantics are harmful *inside the harness encoding* (accuracy 0.6250 / 0.5833
against a 0.75 no-feedback baseline, with 8 and 10 right-to-wrong flips) while production's
own zero-delta write measured faithful (4.29e-06) and has no decision consumer.

**Falsified — do not re-propose without new evidence:** the Tripartite Resonator replacing
the linear operator (0/16 real ARC tasks); the scalar rotor as an answer selector (gauge
no-op); linear least-squares task functors as relational solvers (about +0.05 headroom only);
the structured character codec at scale; the causal-emergence ratchet on the monolithic
operator at production dimension.

**Blocked:** production-dimension planner (absent checkpoint overlay); CUDA verification
(instance terminated, credit 0); production Zone C DSN (no approved credential channel);
official AAII grading (private graders for 75% of the index).

## 5. What is missing to score SOTA on Artificial Analysis

The composite has 10 members and 30/20/30/20 category weights, and three quarters of its
weight is graded externally (Judge/Elo panels, private grading servers, gated datasets).
Map every gap to the pipeline:

| # | Gap | Unlocks | Status / gate |
|---|---|---:|---|
| 1 | **Semantic backbone** — waves carry no word meaning; the text codec is a hash ("cat" vs "dog" scores the same as "cat" vs "quantum") | ~40-50% | `REQUIRES_APPROVAL`: frozen revision-pinned pretrained backbone as an authorized adapter, contamination review, matched ablations |
| 2 | **Decisive egress head** — margins below chance-matched random control | 40% | retrain the linear head on GPU, or train the Hopfield projection; gate = distinct top-1 tokens above floor on N>=100 prompts |
| 3 | **Production text ingress** — char-phasor codec with a live consumer at production dimension, plus long-context handling | 10-40% | BLOCKED on 1 + compute |
| 4 | **REST egress path** — most members are graded over HTTP from generated text | 40% | bounded by 1 + 2 |
| 5 | **Long-horizon agent loop** — plan/act/observe/re-plan over shell tooling | 30% | engineering on top of the existing EFE loop; gate = one objective agentic task end-to-end |
| 6 | **Compute** | all | BLOCKED: funded Blackwell instance (re-scan the market; it rotates) |
| 7 | **Evaluation governance** | prereq | already built (pinned datasets, evaluator digests, item-level outcomes); it has never had a model worth scoring |

**Order:** 6 gates everything; among the rest, 1 → 2 → 3 → 4, with 5 riding along on 1-2.
Gap 1 is the honest hard point and a **deliberate architectural decision**, not a tuning
knob: no amount of wave-mathematics tuning substitutes for grounded world knowledge.

## 6. Standing caution

No harness number in this document is a capability claim. The live external score remains
**0.0%**; the 0.7833 figure quoted elsewhere is a multiple-choice recognizer scoring
pre-built candidates, not a model solving tasks. The sparse-delta, EFE-planned-margin and
demo-path findings are mechanism and diagnostic results, and are labelled as such.

## 7. Measured-evidence addendum -- two probes, and a retraction of my own correction

### 7.1 Probe 1 -- the raw number (`probe_codec_semantics.py`)

The text codec was probed directly (the class was located by scanning repo-root
modules; it lives in `zone_c_epistemic_axiom_harness.py`), five pairs at `D = 65536`:

| pair | raw cosine |
|---|---:|
| `cat` vs `cat` (identity control) | 1.000016 |
| `cat` vs `dog` (related) | 0.748165 |
| `cat` vs `quantum` (unrelated) | 0.747066 |
| `a+b` vs `b+a` (order swap) | 0.749577 |
| `27` vs `28` (adjacent) | 0.749419 |

Read naively: identity is 1.0 (the codec is deterministic) but distinct pairs all sit
near 0.748 regardless of meaning. But 0.748 is suspiciously high and suspiciously
CONSTANT, so the number itself needed auditing before it was trusted.

### 7.2 Probe 2 -- is 0.748 semantics, or a metric artifact? (`probe_metric_artifact.py`)

The codec returns a **uint8 ring on [0, 255]**. Its measured mean is **127.1758**. A
large positive constant present in BOTH vectors forces their raw cosine up regardless
of phase structure. Removing each vector's own mean isolates the part that can carry
meaning:

| pair | raw cosine | **CENTERED** | Z_256 agreement (<=1 step) |
|---|---:|---:|---:|
| `cat` vs `cat` (identity) | 1.000016 | **1.000000** | 1.0000 |
| `cat` vs `dog` (related) | 0.748165 | **0.001112** | 0.0114 |
| `cat` vs `quantum` (unrelated) | 0.747066 | **-0.000938** | 0.0123 |
| `a+b` vs `b+a` (order swap) | 0.749577 | 0.001485 | 0.0116 |
| `27` vs `28` (adjacent) | 0.749419 | 0.002686 | 0.0125 |

Chance rate for Z_256 agreement within 1 step is `3/256 = 0.011719`.

Reading: **0.748 is the DC carrier, not similarity.** Centered, distinct pairs land at
`-0.0009 ... +0.0027` -- i.e. no signal -- and circular agreement is exactly the chance
rate. The discrimination that matters is the centered GAP: 0.001112 vs -0.000938 gives
**0.002050**, which is indistinguishable from zero.

### 7.3 Verdict, and a RETRACTION

**CONFIRMED (the audit's central claim):** the qFHRR text codec carries *identity* but
not *meaning*. Centered gap ~0.002; circular agreement at chance. A related pair is as
similar as an unrelated pair. This is the single largest gap between HENRI and any
language-knowledge composite, and it is now OBSERVED with a reproducible probe.

**RETRACTED -- my own correction was wrong.** An earlier draft of this section asserted
that the architecture catalogue's "*~1/sqrt(D)*" magnitude was `FALSIFIED` and that the
true distinct-pair cosine was ~0.748. **That was false.** Raw cosine (DC-dominated,
~0.748) and centered/ring-aware similarity (approx 0, matching ~1/sqrt(D) ~ 0.0039)
measure **different quantities**; comparing them was a scale/metric conflation of the
same class this project repeatedly catches. The catalogue number is **not** contradicted.
Recording the retraction rather than quietly deleting it, because the naive reading is
attractive and a future reader will meet it again.

**Lesson worth keeping:** the discriminating statistic for a ring codec is the
DC-removed gap, never the raw cosine. Any future "codec is structureless" or "codec is
compositional" claim must state which of the two it measured.

**Probes:** `experiments/verification/probe_codec_semantics.py` (raw view) and
`experiments/verification/probe_metric_artifact.py` (centered + Z_256 view). Both are
read-only, re-runnable, and make no repo writes.

## 8. Commit ledger for this audit window

| commit | content |
|---|---|
| `cd28c7c` | sparse-delta finding (harness-scoped) + demo-path answer-coupling |
| `7eac5cd` | Priority 2: EFE-planned margin at reduced scale + fabrication correction |
| `0fc3df3` | retraction of my own production-defect overclaim |
| `012a023` | removal of the unconditional "Zero-Shot Success" banner |
| this commit | the audit register + the two probes that ground its central claim |

All pushed. Seal gate 7/7 on each. Suite 2106 passed / 21 skipped / 0 failed.

## 9. Scope statement

No harness number in this document is a capability claim. The live external score
remains **0.0%**. The 0.7833 figure quoted elsewhere is a multiple-choice recognizer
scoring pre-built candidates, not a model solving tasks. The sparse-delta,
EFE-planned-margin and demo-path results are mechanism and diagnostic findings.
