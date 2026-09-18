# Underconfident Attractor — Actions 1–5 Outcome Register

Status: `EXECUTED. Actions 1–4 measured; Action 5 honoured (no spend).`
Date: 2026-09-18
Commits: `2dd8073`, `8a3ca69`, `f3c1c92` (all pushed to `origin/main`)
Source: `Project HENRI_ The Underconfident Attractor, Calibration Physics &
Strategic Action Blueprint.pdf` (8 pp.)
Evidence labels: `OBSERVED` · `DERIVED` · `INFERRED` · `HYPOTHESIS` · `FALSIFIED` · `BLOCKED`

---

## Action 1 — Reconcile remote origin: DONE, durable

    To https://github.com/cjc214foodun9/HENRI.git
       2dd8073..8a3ca69  main -> main        PUSH_RC=0
       8a3ca69..f3c1c92  main -> main        PUSH_RC=0

`OBSERVED`: after fetch, `main == origin/main == f3c1c92`, `ahead=0 behind=0`.
The CRLF repairs, the 60-task calibration receipt and the Actions 2–4 work are in
remote versioned history.

---

## Action 2 — Temperature scaling: `MEASURED`, blueprint range `FALSIFIED`

Receipt: `experiments/verification/temperature_scaling_observed.json`
(file sha256 `dbacb29a05fa44ef523ca4aba93d7e78534873aed52918a19176f22ce6ff7711`)
Generator: `experiments/verification/gen_temperature_scaling_receipt.py`
Every number comes from `henri_probe_calibration` functions — no scoring rule is
re-implemented in the generator.

### Self-check (definitions match the sealed receipt)

| Quantity | Recomputed | Sealed receipt |
|---|---|---|
| accuracy | 0.783333 | 0.783333 |
| ECE | 0.482189 | 0.482189 |
| Brier skill | +0.122060 | +0.122060 |

`max |p_here - p_sealed| = 3.42e-08`; tolerance `1e-06` justified in-file by the
float32 (torch) vs float64 (python) provenance of the stored probabilities. The
earlier `1e-9` gate was unjustifiably tight and is corrected, not loosened to pass.

### Fitted optimum (NLL, the strictly proper scoring rule)

    T* = 0.038316      beta* = 26.0985      NLL = 0.526798
    ECE at T* = 0.056101      ECE at T=1.0 = 0.482189      gain = +0.426088
    reachable ECE floor = 0.053623 at T = 0.046357 (beta* = 21.57)

### The blueprint's beta* range does not satisfy its own gate

| beta* | T | ECE | mean peak | BSS | joint gate |
|---|---|---|---|---|---|
| 4.0 | 0.2500 | 0.317577 | 0.4658 | +0.4003 | **FAIL** |
| 8.0 | 0.1250 | 0.186887 | 0.6016 | +0.5388 | **FAIL** |

`FALSIFIED`: the blueprint states beta* in 4–8 aligns peak confidence with the
0.7833 accuracy "collapsing ECE from 0.48 to ≤0.05". Measured ECE over that range
is 0.19–0.32.

Its **reasoning** is however correct and is now quantified: the T that matches
mean peak to accuracy is `T = 0.036188` (`beta* = 27.633`), giving
`ECE = 0.077242`, `mean_peak = 0.7838`. The blueprint's beta* is off by ~4x.

### Binning robustness (the floor is partly a binning artifact — reported, not hidden)

| bins | 5 | 8 | **10** | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| floor ECE | 0.031866 | 0.046554 | **0.053623** | 0.074810 | 0.075729 | 0.109596 |
| meets 0.05? | YES | YES | **no** | no | no | no |

So the 0.05 gate is reachable only at coarser bin counts. At the standard 10 bins
the floor is `0.053623 > 0.05` — the gate is NOT met by pure temperature scaling.

### Held-out result (the headline) — fit on train, report holdout

    T* fitted on 30 train = 0.043782 (beta* = 22.840)

| part | acc | ECE | BSS | mean peak | NLL |
|---|---|---|---|---|---|
| train (in-sample) | 0.7333 | 0.1230 | +0.5249 | 0.7118 | 0.6612 |
| **holdout (headline)** | **0.8333** | **0.0912** | **+0.7318** | 0.8095 | 0.3966 |
| holdout at T=1.0 | 0.8333 | 0.5223 | +0.1477 | 0.3110 | 1.1856 |

40 random 30/30 splits, T* fitted per split by NLL:
held-out ECE mean **0.1301**, min 0.0431, max 0.2099; T* median 0.03832
(range 0.01450–0.06916); **1/40 splits pass the joint gate**.

`INFERRED`: a single split can be lucky. The mean held-out ECE (0.1301) is the
honest expectation, and the in-sample floor (0.0536) is optimistic.

### Correction of my own claim (`FALSIFIED`, recorded in code and tests)

An earlier version of the `brier_skill_score` docstring asserted Brier skill was
**invariant** under temperature scaling. Wrong. Measured: BSS moves
`+0.122060 -> +0.630599`. The multiclass Brier sum contains probability
*magnitudes*, not only the argmax, so a positive logit rescaling changes it.

What IS invariant: **argmax, and therefore accuracy** (analytic — softmax(z/T)
preserves ordering for every T>0). Temperature therefore moves calibration only
and cannot manufacture accuracy.

The false claim was corrected in: the module docstring, the receipt field
(`bss_invariant_holdout` → `bss_improved_holdout`), and the test, which now
asserts the measured behaviour in both directions. The pre-registration is kept
in **inverted** form so it stays falsifiable.

---

## Action 3 — Directive 4 wiring in `production_arc_run.py`

Behind `HENRI_TYPED_PROBE_CONTRACT=1`, **default OFF**; the import block is lazy,
so an unset flag costs nothing and the default path is byte-identical.

### Operator divergence (stated, not silently substituted)

The blueprint writes `Psi_{t+1} = Proj_{S^{D-1}}( Psi_t + K (x) V_obs(dS) )` —
additive bundling then sphere projection. What is **measured and contract-tested**
here is per-dimension **phase rotation** (`apply_wave_binding`). The tested
operator is used; the untested algebra is not.

### The scalar trap, handled explicitly

`dS` is a **scalar**. Broadcast as a scalar rotor it is a U(1) gauge
transformation (measured normalized overlap `0.999999940`) and carries nothing;
quantized into buckets it destroys magnitude. So `dS` is spread across the latent
dims as a **bounded ramp** (`dim i` gets `scale*dS*(i+1)/latent`) plus the action
key's Z_256 phase code. It is never numel-1, so `reject_scalar_rotor` cannot fire
on it, and the ramp **slope** recovers `dS`.

### Anti-leakage

The delta is bound into `probe_belief_wave`, a carrier **separate** from
`state_wave`. `state_wave` feeds `train_ctx`, so binding a post-action
observation into it would let a FUTURE observation justify the PRIOR action's
state. A test asserts the probe channel does not mutate `state_wave`.

### Live falsification

Every active step emits `TYPED_PROBE_TRANSDUCTION` telemetry carrying
`delta_s`, `recovered_delta` and `delta_roundtrip_abs_err`, so the encode is
inverted and checked against itself in the live loop. `ScalarRotorRejected` on
this path raises `SystemExit(BLOCKED)` rather than falling through.

### Real defect found by the new tests, and fixed

`recover_delta_from_wave` (first version) returned **0.3816 for a true delta of
0.0** — a genuine defect, not a tolerance issue. The absolute phase of a wave
contains its own pre-existing structure, so a single post-transduction wave does
not determine the delta. A `reference` wave is now **required**; the
reference-free path is documented `CONDITIONAL` and pinned by a test asserting it
is unreliable.

---

## Action 4 — Adversarial codebook control: `INCONCLUSIVE` on the metric, `FALSIFIED` for the readout hypothesis

Receipt: `experiments/verification/codebook_adversarial_observed.json`
(schema `henri.codebook-adversarial.v2`)
Runner: `experiments/verification/run_codebook_adversarial_control.py`

### Defect in my own first version (recorded in the receipt)

v1 tested `O_VSA_IngressTokenizer`. The 78.33% receipt is produced by
**`HENRIVisionEncoder`** (`run_calibration_eval.py:262`); the tokenizer only
shares a compatible interface. v1 therefore did **not** test the encoder behind
the claim it audited. v2 runs both explicitly. The defect is recorded in the
receipt rather than quietly overwritten.

### Why the obvious control is invalid (and is therefore not the discriminator)

"Freeze codebook M, decode with the same M, require a random Gaussian codebook to
fail cleanly" **cannot discriminate at D = 65,536**: random bound states are
near-orthogonal (earlier R2 probe: max pairwise overlap `7.349e-03` vs chance
`3.906e-03`), so a random codebook decodes its *own* binding fine. That test
measures the encode/decode **channel**, not the representation. Requiring it to
fail would rig the control.

### Rotation-margin statistic — `INCONCLUSIVE` (pre-registered kill fired)

| arm | dim | sim related | sim foreign | margin |
|---|---|---|---|---|
| `HENRIVisionEncoder` (receipt encoder) | 65536 | 0.250641 | 0.010384 | **+0.240257** |
| `O_VSA_IngressTokenizer` | 65536 | 0.656093 | 0.384167 | +0.271926 |
| random projection 1 | 65536 | 0.375497 | 0.141257 | +0.234239 |
| random projection 2 | 65536 | 0.376011 | 0.141470 | +0.234542 |
| random projection 3 | 65536 | 0.375113 | 0.142384 | +0.232729 |

40 real ARC tasks; encoder refusals 0 (`ok=241`). Pre-registered **P2 KILL fired
for both encoders**: a structure-blind random projection reproduces a comparable
margin (`0.234` vs `0.240`). Any reasonable embedding preserves grid-level
rotational autocorrelation, so this statistic is **confounded**. It cannot
attribute the 78.33% to wave-manifold structure. Reported as `INCONCLUSIVE`,
**not** as support.

Notable and worth keeping: the receipt encoder drives *foreign* grids to
**0.0104** — near-orthogonal, hence a genuinely separating basis.

### Sealed-arms control — the readout hypothesis is `FALSIFIED`

The sealed receipt already contains a **valid** control for the question asked:
its three arms share an **identical readout** (cosine against the same candidate
wave set) and differ only in the **operator**.

| arm | accuracy | ECE | BSS |
|---|---|---|---|
| functor (fitted W) | **0.7833** | 0.4822 | +0.1221 |
| identity (no operator) | 0.7333 | 0.4249 | +0.1337 |
| random (W ~ N(0,I)) | **0.3000** | 0.0487 | −0.0001 |

chance = 0.2500 · random excess over chance = **+0.0500** ·
readout explains the accuracy? **False** · encoder-only accuracy (no operator
fit) = **0.7333** · operator gain over encoding alone = **+0.0500**

`DERIVED`: with the same readout and the same candidates, a random Gaussian
operator scores at chance while the fitted operator scores 0.7833. The readout
projection does **not** manufacture the discrimination. This satisfies the
blueprint's literal requirement — "an identical test initialized with a random
Gaussian codebook fails cleanly" — because `W_rand = torch.randn` **is**
`M_rand ~ N(0,I)`, and it does fail cleanly (0.3000, BSS −0.0001).

**What the blueprint's stronger conclusion does NOT get.** Encoding alone, with
**no operator fit**, already reaches **0.7333**. The fitted operator adds only
**+0.0500**. So "the 78.33% discriminative capability resides within the
algebraic structure of the wave manifold" is only partially supported: the
refuted half is the readout-artifact hypothesis; the affirmed half must be
attributed mostly to the **encoder + candidate construction**, not to the learned
operator or the binding algebra.

---

## Action 5 — Compute discipline: honoured

`OBSERVED`: CPU only, **no spend**. Vast instance `50797414` is EXITED with zero
credit and SSH refuses; Zone C `:10100` is closed. Dependency probe on the local
host: `arc_agi` importable, `arcengine 0.1.0`, `torch 2.11.0+cu128` — but
`torch.cuda.is_available() = False`.

---

## Verification ledger

| Check | Result |
|---|---|
| Targeted suites (temp scaling, transduction, wiring, calibration, probe contract) | **101 passed** |
| Full suite on the changed tree | **2007 passed, 21 skipped, 0 failed** |
| Pre-commit seal gate | **PASS — 7 sealed pairs** |
| Dependent-pin gate | **PASS — 1 set, 0 failed** |
| `production_arc_run.py` | compiles; flag-ON import block resolves |
| Push | `2dd8073..8a3ca69`, then `8a3ca69..f3c1c92`; `origin/main == main` |
| Byte-pinned K3 artifacts | prereg `841ac581…` = pin; kernel `bff01749…` = pin |

Count reconciliation: 1962 (pre-existing) + 45 (new: 20 temp scaling, 18
transduction, 7 wiring) = 2007.

---

## Open items (numbered, bounded)

1. **Action 3 is wired and unit-verified but NOT live-verified.** No ARC
   environment was run: no scorecard, no task score, no WIN. `BLOCKED` on
   compute. Action 5's own gate (fund Vast only after a non-zero live score)
   therefore requires a CPU-runnable end-to-end harness first.
2. **No clean discriminator yet exists** for "structure in the wave manifold".
   The rotation-margin statistic is confounded; the sealed-arms control refutes
   only the readout hypothesis. The next candidate is a *held-out* test in which
   the codebook is frozen before evaluation and a mismatched codebook is required
   to fail — with the failure criterion pre-registered in advance.
3. **The fitted operator contributes only +0.05 over encoding alone.** That is
   the most actionable number here: effort spent on the operator algebra is
   bounded by that headroom until the encoder+candidate pipeline is re-examined.
4. **The 0.05 ECE gate is not met** by temperature scaling at 10 bins (floor
   0.0536; held-out mean 0.1301). Meeting it needs a calibrated head or a
   different readout, not a different beta*.
