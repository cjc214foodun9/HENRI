# Natural-Intelligence Audit → Tree Merge Register

Status: `EXECUTED. 1 wired+tested, 1 REFUTED by live measurement, 1 VOID (self-gated), 1 encoded, 2 recorded.`
Source: `Downloads/project_henri_natural_intelligence_audit_morphogenetic_wave_architecture.md`
(`HENRI-ARCH-2026-NATURAL-INTELLIGENCE-AUDIT`)
Base: `main @ 8114397` → this commit
Date: 2026-09-18

---

## 0. Provenance: the document is downstream of the receipts

The attachment carries identifiers and values first produced **by this project**, after
the artefacts: `T* = 0.038316`, `β* = 26.10`, `+0.6306`, `0.4822 → 0.0536`, Seal Pair 7,
`ScalarRotorRejected`, `probe_belief_wave`, the 16/16 recovery figure, and the 0.0 %
score. It is a **synthesis of** the receipts, not an authority over them. Every claim
below was checked against the artefact that produced it; the `/arxiv` layer was verified
against published sources. Nothing was adopted on the document's authority.

---

## 1. Literature layer — REAL frameworks (`DERIVED`)

Verified via search against primary sources. All five characterise correctly:

| id | framework | verified source | status |
|---|---|---|---|
| L1-TAME | scale-free basal cognition; cognitive light cones; setpoint navigation | Levin, *Technological Approach to Mind Everywhere (TAME)* — arXiv **2201.10346**; Front. Syst. Neurosci. 2022 | `REAL_PUBLISHED_FRAMEWORK` |
| L2-KURAMOTO | phase synchronisation; order parameter `r e^{iψ} = (1/N) Σ e^{iθ_j}` | Kuramoto 1975; Strogatz, *Sync*; Phys. Rev. E **107**, 044211 | `REAL_PUBLISHED_FRAMEWORK` |
| L3-FEP | variational free energy; perceptual vs active inference; Markov blanket | Friston 2010 and later active-inference reviews | `REAL_PUBLISHED_FRAMEWORK` |
| L4-LANGEVIN | SGLD / Langevin creep; Fokker–Planck stationarity | standard non-equilibrium statistical mechanics | `REAL_MATHEMATICS` |
| L5-NORM | Hermitian generator ⇒ unitary flow ⇒ `d/dt‖Ψ‖ = 0` | elementary linear algebra / QM | `REAL_AND_TRUE` |

**Consequence, stated so it cannot be over-read:** citing real frameworks validates the
**framing only**. It is not evidence about this codebase, and it is not a capability
result.

---

## 2. Isomorphism layer — DESIGN METAPHOR, not mechanism (`INFERRED`)

The audit's 1:1 ledger maps biology onto software. The software implements **wave states,
phase vetoes and SGLD creep**. It does **not** implement membrane potentials, gap
junctions or apoptosis. In-tree probe (occurrences in non-archive `.py`):

| named concept | hits | meaning |
|---|---|---|
| `sagnac_veto` | 160 | implemented |
| `stiefel_retraction` | 272 | implemented |
| `hopfield` snap | 273 | implemented (sealed codebook) |
| `sgld` creep | 39 | implemented |
| `kuramoto_order_parameter` | 227 | implemented |
| `membrane_potential` | 6 | **named only** — no membrane is simulated |
| `gap_junction` | 11 | **named only** |
| `apoptosis` | 6 | **named only** |
| `batio3` | 11 | **named only** — no photonic substrate |
| `pockels` | 6 | **named only** |

**Verdict:** rhetorically isomorphic, mechanistically **analogical**. Recorded as
`DESIGN METAPHOR`. It must never be reported as a verified biology-to-physics
translation.

---

## 3. Theorem layer — one trivial, one not proven as stated (`FALSIFIED` as proof)

**Norm conservation.** `H† = H ⇒ U = exp(-iHt/ħ)` unitary ⇒ `d/dt‖Ψ‖ = 0`. The proof is
correct but **trivial**: it follows from Hermiticity by construction and establishes no
property of any learned operator. It cannot be cited as evidence of capability.

**Convergence "theorem".** The Fokker–Planck stationary form
`P∞ ∝ exp(-F/k_BT)` requires conditions (sufficiently mixing noise, an appropriate
temperature schedule) that the document **asserts, not establishes**. More decisively:

> `T_eff → 0 ⟺ Δ_Sagnac ≤ 0.0431`

is a **definition written into the document**, not a derivation. The document defines
`T(Δ_Sagnac) = T_floor + α·Θ(Δ_Sagnac − τ_veto)·Δ_Sagnac` and then reports the
consequence of its own definition. Recorded as *"stationary-distribution form under
standard SGLD assumptions"* — **never** as a convergence guarantee.

---

## 4. The four "resolved" boundary defects — checked against receipts

| # | document's claim | verdict | evidence |
|---|---|---|---|
| 1 | Ingress used degenerate same-sum frequencies; "sealed, 16/16 exact recovery" | **`TRUE`** | `phase_map_basis_observed.json`; Seal Pair 7 |
| 2 | Linear LS "replaced by" the Tripartite Resonator; UWSH ceiling broken | **`REFUTED` — do not ship** | resonator **0/16** real ARC tasks vs incumbent linear **+0.395123**; treatment **+0.144086**; shuffled control wins 7/16 |
| 3 | Scalar rotor → dimensional phase ramps; recovery `< 1e-6` | **`TRUE`** | `henri_underconfident_attractor_actions.md`; round-trip 4.29e-06, channel hit 1.000; wired behind `HENRI_TYPED_PROBE_CONTRACT` |
| 4 | A2 32k head → Modern Hopfield snap; BSS `+0.6306` | **`PARTIAL` + one conflation** | sealed codebook 32000/32000; **action-vocab instance measured near-uniform** (§5 below); the 32k **lexical** path remains `RETRAIN_REQUIRED` |

### Two conflations, refused explicitly

**(a) `+0.6306` / `T* = 0.038316` belong to a DIFFERENT readout.** They were fitted on the
**functor-score** readout over the 60-task calibration receipt. The document attaches them
to the **Hopfield snap**, whose calibration is **unmeasured**. Transferring a fitted
temperature across readouts without measuring the new one is precisely the defect class
this project keeps catching. No such transfer is made here.

**(b) "The low score was not an ontological failure of the wave core" is half-right.**
The encoder alone carries **0.7333** of the 0.7833, so the core is real. But the
document's *prescription* for defect 2 is measured **worse** than what it would replace.
Honest form: the transmission was uncoupled **and** the proposed operator swap is a
measured regression.

---

## 5. `top1_margin` on LIVE ENCODER WAVES — decisive negative (`OBSERVED`)

Action 2 asked for the margin on a flag-ON episode. A full episode needs live ARC
environments; the cheaper **honest** source is the production ingress over the local ARC
corpus — real grids through `HENRIVisionEncoder`, not isotropic noise. Runner:
`run_sealed_margin_live.py`; receipt `sealed_margin_live_observed.json`
(`fd65518ac89a96e8…`).

**n = 1026 waves** (real ARC training tasks), **0 encode/decode failures**, 8 of 8
actions covered:

| statistic | LIVE encoder waves | random waves (prior) |
|---|---|---|
| mean margin | **0.023355** | 0.050150 |
| margin p1 / p5 / p10 | 0.000135 / 0.000865 / 0.001432 | — |
| margin p50 / p90 / p99 | 0.010198 / 0.049692 / 0.259115 | — |
| mean normalized entropy | **0.980811** | 0.974660 |
| difference (live − random) | **−0.026795** | — |

**The live readout is LESS decisive than noise.** Lower mean margin **and** higher
normalized entropy than the random-wave baseline. At a floor of 0.02 it would abstain on
**70.37 %** of live waves.

**Decision: NO floor is set, and none is recommended.** Setting `0.02` from the random
distribution would imply the readout works; the live measurement says it does not.
`HENRI_SEALED_EGRESS_MIN_MARGIN` therefore remains `0.0` with its existing loud `[init]`
warning, and `HENRI_SEALED_ACTION_EGRESS` stays **default-OFF**. The p05 (0.000865) and
p10 (0.001432) figures are recorded as *descriptive of abstention tolerance only* — they
would abstain on 5 % / 10 % of live waves and carry **no** accuracy implication.

**Scope limits (in the receipt):** encoder waves, **not** EFE-planned candidate waves; no
task attempted or solved; no ground-truth action exists for an arbitrary ARC grid, so
**correctness is not measured**.

---

## 6. Sparse-Δ closed loop — `VOID` by its own validity gate (`OBSERVED`)

Action 3 asked for the pillar-1 loop under scorecard sparsity (mostly Δ = 0). Runner
`run_sparse_delta_loop.py`; receipt `sparse_delta_closed_loop_observed.json`
(`f4e8dd4c0c4fb0ef…`).

**Result: `VOID_DOES_NOT_REPRODUCE_DENSE_BASELINE`.** My harness fails its own
pre-registered validity gate:

* **V1 — dense baseline NOT reproduced.** At `p(nonzero) = 1.0` it gives
  change_rate **0.0000** / accuracy **0.6833**, against the pillar-1 receipt's
  **0.1944** / **0.8125**. If p = 1.0 does not reproduce the dense loop, this harness is
  **not the same loop** and no result from it may be read as a sparsity bound on that loop.
* **V2 — the retrieved sign is DEGENERATE.** `positive_frac = 1.000`: every retrieved
  logit was positive, so the "signed" arm is byte-identical to the positive-only arm and
  measures nothing.

**Consequence, recorded rather than hidden:** P1 and P2 "pass" **vacuously** — "change
rate is lower than dense" is automatic when the change rate is 0.0 for a degenerate
reason. The receipt's `pre_registered.P1_P2_status` says so in those words, and the
`non_claims` block states the run is not interpretable as a sparsity result.

**What the run does establish (`P4`, informative regardless):** the wrong-sign arm moved
**0.7167** of decisions and dropped accuracy **0.5375 → 0.4208** with **82** right→wrong
flips. A sign error actively destroys performance. That is a real control result.

**Still open:** the sparse-Δ question is **unanswered**. It requires extending the
**verified** pillar-1 harness (`run_closed_loop_microharness.py`) — which reproduces
0.1944 — rather than a reimplementation.

### Two defects in my own harness, found by running it

1. **Shape bug:** the ramp was built 256 long and reshaped to `(64, 8)` = 512.
2. **Design bug (the important one):** the first version biased a **per-action vector**,
   which is not monotone in the incumbent argmax, so `P3` (positive-only ⇒ change_rate
   exactly 0.0) could not hold. The pillar-1 result depends on boosting the **current
   choice**; the arm now biases the incumbent only, which makes P3 falsifiable rather than
   accidental.

---

## 7. `receipt_path.py` — wired, tested, and the committed artifacts PROVEN untouched (`OBSERVED`)

Action 1. Contract: priority `--out` > `HENRI_RECEIPT_DIR` > **default byte-identical**;
a bad override **raises `ReceiptPathError`** rather than falling back to the ledger-cited
path, because silently writing the committed location is the exact failure this prevents.
Wired into all three runners (`run_egress_snap_verification.py`, `run_trilevel_loop.py`,
`run_hardware_blocked_register.py`), each printing a `REDIRECTED` notice so a diverted run
is visible in its own log.

**Measured (smoke test, `--out` into a temp dir):**

| runner | rc | redirect notice | committed receipt |
|---|---|---|---|
| `run_hardware_blocked_register.py` | 0 | True | **UNCHANGED** |
| `run_egress_snap_verification.py` | 0 | True | **UNCHANGED** |
| `run_trilevel_loop.py` | 0 | True | **UNCHANGED** |

All three landed in the temp dir; bad overrides raise. `tests/contract/test_receipt_path.py`
**20/20** includes subprocess integration that hashes the committed receipt before/after.
**No verification was run against a committed path.**

### Two test defects of my own, caught by the tests

1. **Prose false positive:** the CI-order check matched `--check` inside the workflow's
   own explanatory **comment**, reporting `1543 < 370`. Fixed by stripping comments; a
   negative-control test now pins that a comment cannot satisfy the check. Same class as
   the earlier "reflex fastest" and dead-flag false positives.
2. **Window bug:** an 80-character window for the "bare call" match spilled into the next
   line and swallowed the following `--check`, so every call looked like a check call.
   Line-bounded.

---

## 8. CI ordering — encoded, not assumed (`OBSERVED`)

Action 4. `.github/workflows/egress-manifest.yml` **regenerates the derived manifest
before `--check`**. This matters because the pin states the manifest is derived and
gitignored, and that a consumer **must fail closed when absent** — so on a clean clone
`--check` returns `CHECK_FAIL` on a **healthy** tree (observed: `CHECK_FAIL`, then `PASS`
after regeneration).

The workflow **fails loudly** (`exit 1` + `::error::`) when the source tokenizer artifact
is absent, because a green run with an unregenerated manifest would be a false signal about
the seal. `tests/contract/test_ci_manifest_ordering.py` **7/7** pins the order,
comment-aware, with a negative control proving a comment cannot satisfy the check.

**Trigger is manual dispatch, deliberately.** A push-triggered version would run on every
commit and fail on any runner lacking the source `tokenizer.json` — painting `main` red for
an **environmental** reason, which is a worse signal than an explicit gate. Regeneration
cannot be faked: without the source artifact there is nothing to verify against the pin. To
automate it, provide the artifact to the runner and add a `push:` trigger. **No CI runner was
executed here** — the tests pin the recipe, not a run.

---

## 9. Open register items (recorded, with owners)

1. **Demo-pair SGLD loss direction — run-dependent, predominantly RISING.** Three
   observations: 10.393597 → 10.457414; 10.396907 → 10.457165; 10.400698 → 10.445483. One
   falling case (10.395647 → 10.395215) shows the direction is not systematic. The banner
   prints `[Phase C Zero-Shot Success]` while the loss rises, so **"success" is not a
   supported label** for that path. Owner needed: either the adaptation is mis-scaled or
   the banner is unsupported. Cost: **~60 s/call** at D=512 (three measurements:
   60,331.06 / 60,676 / 60,412.62 ms) — **slow, not hung**.
2. **`HENRI_SEMANTIC_EGRESS` is a DEAD STORE** — 2 occurrences repo-wide: its own
   definition and my explanatory comment; **zero reads**. Deliberately **NOT repurposed**
   by the sealed path: its documented target is the 32k **code-token** decoder (defect A2),
   a **different vocabulary** from the 8-action set.
3. **Hardware chapter** — `claimed_substrate_figures`: **11 claims, 11 BLOCKED**;
   `energy_per_step = None` by construction. The `< 6.67 nJ/step` and `< 0.1 ns transit`
   figures are **UNFALSIFIABLE on this host** and are neither confirmed nor denied.
4. **Repo-wide `eol=lf`** and **Rust/CXL/photonics scaffolding** remain **WITHHELD**
   pending explicit approval.

---

## 10. Non-claims

* **Live benchmark score remains 0.0 %.** No ARC task was attempted or solved by anything
  in this change set.
* The 0.7833 figure is a multiple-choice **recognizer** over **pre-built** candidates; the
  operator adds only **+0.05** over the encoder, so no operator claim follows from it.
* The sealed **action** readout is **less decisive than noise** on live encoder waves and
  therefore must not be used as a policy readout.
* The sparse-Δ question is **unanswered** (harness `VOID`).
* No biology-to-physics translation is claimed; the isomorphism is a **design metaphor**.
* No substrate figure is promoted to measurement; no "free lunch" claim survives.
* `ECE ≤ 0.05` remains **retired** and pinned by tests (10-bin in-sample floor 0.0536;
  held-out mean 0.1301).
