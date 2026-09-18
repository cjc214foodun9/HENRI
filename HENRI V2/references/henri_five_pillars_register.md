# Five Pillars — Outcome Register

Status: `EXECUTED. 1 demonstrated, 1 partial, 1 verified, 1 REFUTED, 1 BLOCKED.`
Date: 2026-09-18
Base: `main @ 25d80f1` → this commit
Source: `Project HENRI_ Architectural Evaluation of Underconfidence, Roadmap, and
Foundational Completeness.pdf` (15 pp., ID `HENRI-ARCH-2026-UNDERCONFIDENCE-EVAL-AND-ROADMAP`)
Evidence labels: `OBSERVED` · `DERIVED` · `INFERRED` · `HYPOTHESIS` · `FALSIFIED` · `BLOCKED`

## Provenance note (read first)

The PDF contains names first coined in THIS project's prior session — `ScalarRotorRejected`
(×2), `probe_belief_wave` (×2), `henri_probe_calibration` (×1) — and my measured numbers
(`0.6306`, `0.0536`). It is therefore **downstream of the artifacts it evaluates**, not an
independent authority over them. Every claim below is checked against live code and
measurement; none is accepted because the document asserts it. Marker counts for the
session-authored names are in the extraction log; `S1.1`, `8.4` and `5.3` are absent (0).

---

## Immediate engineering decisions

### D1 — Accept T\* = 0.038316 (β\* = 26.10): ACCEPTED as a fitted parameter

`OBSERVED` in commit `f3c1c92`. Implemented as `FITTED_TEMPERATURE_60 = 0.038316`
(`FITTED_BETA_60 = 26.0985`) with a single resolution site,
`resolve_readout_temperature`. Default remains **1.0**, so production stays
byte-identical; a consumer must opt in via `HENRI_READOUT_TEMPERATURE=fit` or an
explicit float. Bad values **fail closed** (raise), because silently falling back
to 1.0 on a typo is the dead-store defect this project has already been bitten by.

32 tests cover this path, including 11 parametrised bad-value rejections.

### D2 — Retire the ECE ≤ 0.05 gate via temperature: ACCEPTED, gate RETIRED

`DERIVED`. The bound is analytic: as T → 0 every prediction lands in the top
confidence bin, so `ECE → |accuracy − mean_confidence| → 1 − accuracy = 0.2167`.
The measured 10-bin floor is **0.053623** (T = 0.046357); it is reachable only at
coarser binning (5 bins → 0.031866, 8 → 0.046554), which changes the metric, not
the model. The retirement is recorded in code **and pinned by tests**
(`test_ece_gate_is_recorded_as_retired_and_unreachable`), so a later session
cannot quietly resume chasing 0.05.

**Not claimed:** that the result passes the gate. At T\* the full-60 ECE is
0.056101; the honest held-out expectation is the **40-split mean 0.1301**, not the
favourable single-split 0.0912. A test asserts the single-split figure is never
used alone.

---

## Pillar 1 — Closed-loop causal agency: `DEMONSTRATED` (channel + coupling)

Harness: `experiments/verification/run_closed_loop_microharness.py`
Receipt: `closed_loop_microharness_observed.json` (sha `44b9127d6c92`)

The prior commit left the carrier written but with **zero live score**. This
harness closes the loop on 12 real ARC tasks × 4 steps, CPU, using the production
`probe_from_logits`, `transduce_external_outcome`, `recover_delta_from_wave`,
`bind_key_value_wave` and `retrieve_value_from_binding`.

| arm | channel hit | decision change | acc with bias | acc without |
|---|---|---|---|---|
| **signed feedback** | **1.000** | **0.1944** | **0.8125** | 0.7500 |
| reinforce-only (v1) | 1.000 | **0.0000** | 0.7500 | 0.7500 |
| random composite (control) | 0.111 (chance 0.125) | 0.3889 | 0.5833 | 0.7500 |

All invariants held: no spurious `ScalarRotorRejected`, `max |round-trip error| = 4.292e-06`
(bound 1e-3). Flips: **3 wrong→right, 0 right→wrong**.

### Defect in my own v1, and the mechanism it exposed

v1 bound the previous **choice** and applied an **always-positive** bias → 
`decision_change_rate = 0.0000`, verdict `VOID_NO_COUPLING`. That verdict was a
correct measurement of a broken DESIGN, not of the carrier:

> **Argmax monotonicity:** adding `c > 0` to the entry that is *already* the
> argmax cannot change the argmax unless the vector is degenerate. The score
> vector is computed once per task, so it is constant across steps. The bias was
> therefore *structurally incapable* of changing any decision.

That is a **confirmation-bias loop**: it reinforces what it already believed and
can only report agreement — the same defect family as the scalar gauge rotor,
reached by a different route. v1 is retained as the `reinforce_only` arm precisely
because the negative is informative: a naive closed loop is a no-op **by
construction**. The corrected loop bounds the **composite** `choice*2 + Δ` and
applies a **signed** bias, which can flip an argmax.

### Honest limits (in the receipt, not buried here)

- **Dense-delta caveat:** Δ here is dense (every candidate carries a corpus
  label), so live ARC-AGI-3 scorecard deltas — mostly 0 — would give a **lower**
  correction rate. The mechanism is demonstrated; the **rate is not transferable**.
- The improvement comes from **suppressing a verified-wrong choice** and
  re-argmaxing to the next-best candidate. That is genuine error correction, but
  it injects no information about which candidate is *right* — only which was wrong.
- **Not a benchmark score.** The agent selects among **pre-built** candidates and
  solves no task. Live task score remains **0.0%**.

---

## Pillar 2 — Non-linear program synthesis: `REFUTED` (replacement not justified)

Runner: `experiments/verification/run_resonator_vs_linear.py`
Receipt: `resonator_vs_linear_observed.json` (sha `52f25436691d`)

**Finding 1 — the resonator already exists.** `arc_tripartite_resonator.py` (87 KB)
is in the tree, with `scene_real_arc`, its own status constants
(`VOID_CONTROL_NOT_SEPARATED`, `VOID_CALIBRATION_VACUOUS`, `VOID_NON_CONVERGENT`)
and an explicit non-claim: *"The synthetic scenes are solvable BY CONSTRUCTION and
prove nothing about real ARC."* So the action item's "build it" framing was stale,
and no code was greenfielded.

**Finding 2 — the premise is also stale.** The action item says to *"replace linear
circular correlation unbinding."* `arc_task_functor.py` already documents that this
was tested and discarded: the FFT/circulant form scores **≈0.00** on the production
`[num_blocks, 8]` layout because it circulates the *block* axis, whereas a spatial
transform lives in the per-(block, slot) phase. The live default is per-slot
diagonal ridge LS. **There is no linear circular-correlation path left to replace.**

**Finding 3 — the swap fails on real data.** 16 real ARC tasks, the module's own
`evaluate_arms`, all arms on one readout and one hold-out rule:

| arm | mean held-out cos |
|---|---|
| **treatment (tripartite)** | **+0.144086** |
| **control_diag_ls (incumbent linear)** | **+0.395123** |
| control_identity | +0.226541 |
| control_shuffled | +0.117170 |
| control_random_direction | +0.000155 |

**treatment − diag_ls = −0.251037. Beats diag_ls on 0/16 tasks. Beats identity on
5/16.** This reproduces the module's own single-task selfcheck (0.2686 vs 0.6608)
on a broader sample.

**Verdict: `PILLAR_2_REPLACEMENT_NOT_JUSTIFIED`.** The incumbent linear operator
matches or beats the tripartite resonator on a **majority** of real ARC tasks. The
proposed swap is refuted for this data and must not ship as an improvement.

**What this does and does not refute.** The rationale given — *"an operator that
adds only +0.05 over the raw input cannot solve relational logic; relational
reasoning requires discrete combinatorial branching"* — remains an untested
`HYPOTHESIS` about a *different* operator family. What is refuted is narrower and
harder: **this** resonator, as implemented, is not a better operator than the
linear baseline on real ARC waves. The +0.05 headroom finding stands, and it is
the ceiling any operator work must clear.

---

## Pillar 3 — Cryptographically sealed egress grounding: `VERIFIED` (regenerated + pinned)

`OBSERVED`. I regenerated the derived manifest from its source and checked it
against the committed pin:

```
bytes = 242918   tokens = 32000   separator = U+000A '\n'
sha256        = 0f97b4337921e6e7e9b4620fc73338ee570aecd3c16038bc23870a887e995045
pin sha256    = 0f97b4337921e6e7e9b4620fc73338ee570aecd3c16038bc23870a887e995045
manifest_len_matches_cfg = True     CHECK = PASS   (rc=0)
```

Source: `EleutherAI/llemma_7b` `tokenizer.json` (snapshot `e223eee41c…`, present in
the local HF cache); ids 32000–32015 are 16 appended specials, so the manifest
carries exactly 32,000 lexical entries.

**Governance point, verified rather than assumed:** the canonical manifest is
**derived and gitignored** (`.gitignore:39 HENRI V2/data/`) and the pin file was
committed at `ece2710`. The pin's own contract states the file is deliberately not
committed and that a consumer **must fail closed when it is absent** — which is
exactly what `--check` did before I regenerated it (`CHECK_FAIL manifest missing`).
So the earlier `CHECK_FAIL` was **correct behaviour**, not a defect, and the seal
chain is intact: builder + pin committed, artifact regenerable, digest checkable
from a clean clone.

Also `OBSERVED`: seal gate **PASS — 7 sealed pairs consistent**; the builder's own
wiring note confirms the manifest does **not** make generation work (A2 stays
`RETRAIN_REQUIRED`).

---

## Pillar 4 — OOD generalization under embargo: `PARTIAL`, with the scope corrected

Runner: `experiments/verification/run_embargo_ood.py`
Receipt: `embargo_ood_observed.json` (sha `bbcb0b309408`)

**Scope correction (stated, not glossed).** ARC tasks carry **no timestamp**, so
there is no time axis to embargo. This harness implements **task-disjoint OOD**:
a single batch operator is fitted on train task IDs only and scored on disjoint
IDs. Calling that "temporal" would be a labelling upgrade the data does not
support, so the receipt says `TASK-DISJOINT OOD, not a time-series embargo`.

Split frozen before scoring and recorded (`seed 20260918`, 12 train / 12 embargoed,
IDs in the receipt), so the split cannot be re-chosen after seeing results.

| split | operator | identity | random op |
|---|---|---|---|
| in-sample (train IDs) | 0.8333 | 0.8333 | 0.3333 |
| **embargoed (unseen IDs)** | **0.7500** | 0.5833 | 0.2500 |

`operator gain over identity`, embargoed = **+0.1667**; chance = 0.25.
The random-operator control stayed at chance, so the fit carries real task
information. Verdict: **`PILLAR_4_PARTIAL`** — the batch operator generalizes
**above chance** to unseen task identities and beats identity, but the sample is
12 embargoed tasks and the gain is small. That is the number; it is not a
capability claim.

---

## Pillar 5 — Substrate thermodynamic advantage: `BLOCKED`

Runner: `experiments/verification/run_substrate_thermodynamic.py`
Receipt: `substrate_thermodynamic_observed.json` (sha `e3940d33b73a`)

Inventory (`OBSERVED`): `torch 2.11.0+cu128`, `cuda_available = False`,
`cuda_device_count = 0`, `directml_available = False`, `mps = False`, `cpu_count = 16`,
`nvidia-smi` present but **`rc=4`** (no driver-visible GPU), no portable CPU power
sensor API on this host.

`energy_measurable = False` ⇒ **`advantage = None`**, by construction. The energy
denominator of `joules / correct_answers` cannot be measured, so **no advantage is
claimed and none is denied** — absence of measurement is not a result. A CPU
workload proxy (`600 × 512×512 matmul`, 0.3869 s) is recorded **as time, explicitly
not energy**: dividing it by an assumed wattage would fabricate a result.

`VERDICT: BLOCKED`. The protocol to run when a sensor exists is written into the
receipt (fix task + baseline, measure joules with a real sensor, measure
correct answers, compute the ratio, report it with its sensor and interval, claim
nothing at ratio ≥ 1.0 or when the baseline is absent).

---

## Verification ledger

| Check | Result |
|---|---|
| Full suite | **2039 passed, 21 skipped, 0 failed** |
| New tests this commit | 32 (temperature acceptance) |
| Seal gate | **PASS — 7 sealed pairs** |
| `production_arc_run.py` | compiles |
| Egress manifest | regenerate + `--check` **PASS** vs pin `0f97b433…` |
| Five receipts | all written, shas in this register |

Count reconciliation: 2007 (prior HEAD) + 32 = 2039.

---

## The live score, stated plainly

**The live benchmark score remains 0.0%.** The 0.7833 figure is a **multiple-choice
recognizer** choosing among pre-built candidates; this register adds a closed-loop
channel and a task-disjoint OOD measurement, and neither of those is a task score.
HENRI holds a verified engine on a dyno and still lacks the transmission.

## Open items (numbered, bounded)

1. **Pillar 5 is unmet and hardware-bound.** No substrate claim is possible until a
   joule-measuring substrate exists; the blocker is budget, not method.
2. **Pillar 2's next candidate is unconstrained.** The replacement is refuted; the
   +0.05 operator headroom remains the ceiling any future operator must clear.
3. **Pillar 1's rate is not transferable** to sparse production deltas. The next
   honest step is a loop driven by a **sparse** scorecard signal, where most steps
   carry Δ = 0.
4. **Pillar 3 is intact but unrehearsed in CI.** The manifest is derived, so a clean
   clone must regenerate it before any check; that ordering belongs in CI.
