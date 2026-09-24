# UHR-05 Task 7 — Autoresearch reconciliation: what the literature says about the defects I measured

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Evidence classes:** `OBSERVED` (own read/exec) · `DERIVED` (rule shown) · `INFERRED` · `HYPOTHESIS` · `BLOCKED`
**Sources:** `HENRI V2/.ar/` (main checkout), `orx 0.2.10`, all receipts `status=pass`

---

## 0. Status of the two halves — read this first

| Half | Status | Reason |
|---|---|---|
| **Contingency / attribution methods** | `OBSERVED` — grounded | 14 orx receipts, 93 candidates, abstracts read from `.ar/*.orx.json` |
| **Philosophy (Hume / McGilchrist)** | **`BLOCKED`** | NotebookLM auth **expired**; `refresh_auth` → `reason: stale`. No citation fabricated. |

The philosophy half is blocked on a credential the arbiter cannot obtain:
`nlm login --manual --file <cookies>` needs a cookie export, and bare `nlm login`
opens an interactive Google session. **User action required.** Per the ontology
skill's rule: *"If auth is stale, record `BLOCKED` and do not fabricate a citation."*

What **is** already in the record is live code, not a corpus answer — quoted verbatim
so it is not confused with one:

`henri_causal_contingency.py:18-23` (`OBSERVED`):
> **HUME'S CRITERION.** Constant conjunction requires more than temporal correlation.
> Two events being adjacent in time (and even repeatedly so) does not license "cause";
> Hume's own examples turn on the conjunction being between a candidate cause and an
> effect that varies WITH it. A regular beat that advances independently of the action
> is not a conjunction between the action and the world, it is a conjunction between
> the action SCHEDULE and an exogenous clock.

`henri_causal_contingency.py:34` (`OBSERVED`):
> nonzero-change detector is **necessary and NOT sufficient**.

That is the **implemented** criterion. The corpus has not yet been asked whether it
agrees, and no answer is claimed.

---

## 1. Receipt census (`OBSERVED`)

```
receipts.jsonl : 14 rows   kind={query:2, citation:12}   status={pass:14}
                 tool={orx:14}   schema_version={1.0:14}
                 created_at span 2026-09-24T17:51:39 .. 2026-09-24T17:51:39
result files   : 14 *.orx.json (7 keyword + 7 embedding), 93 unique candidates
```

The two `query` receipts (`orx discover keyword`, `orx discover embedding`) each
returned **6 candidates**. The 12 `citation` receipts each carry
`"RETRIEVAL ONLY: this receipt does not ..."` — a receipt proves **what was
fetched**, never what is true.

**Correction to my own earlier report.** I first reported "6 cited IDs have no
persisted candidate record". That was a **sampling artifact of mine**: I had read
`items[:3]` of each list (40 of 93 candidates), so IDs alphabetically later looked
absent. Reading every element, **0 of 12** are missing. No provenance gap exists.

---

## 2. The load-bearing finding: independent corroboration of measured defects

This is not a literature list. Four of the seven defects I found by executing HENRI
code this session are **independently described in the retrieved literature**. That
is corroboration of the *mechanism*, not of HENRI's capability.

### 2.1 Common-mode domination → subtract the difference (strongest)

**Measured (`OBSERVED`):** `fractional_shift` egress — mean pairwise `|cos|` among
encoded prompt waves `= 0.842`; mean `|cos(w, mean_w)| = 0.8421`, i.e. ~71 % of wave
energy points along **one shared direction**. Argmax-unique rose 17/40 → 30/40 when
the query's common mode was removed.

**Literature (arXiv 2609.09902, *Contrastive Projection*), abstract `OBSERVED`:**
> Reading a transformer's internal states in token space is easy to do and hard to
> trust: a logit lens on a single hidden state is dominated, at intermediate layers,
> by the generic tokens the model would predict for almost any input. We read the
> **difference** instead. Subtracting two closely matched prompts' hidden states and
> projecting through the unembedding **cancels the shared component** and surfaces
> what separates them.

`DERIVED`: the mechanism I patched (`remove_common_mode`, flag-gated default-OFF) is
the same operation the paper reports as necessary. My finding was a common mode
carrying ~71 % of the energy; the paper reports a shared component swamping the
signal and prescribes cancellation by differencing. **Independent arrival at the same
remedy.**

### 2.2 A same-construction random control can retain alignment

**Measured (`OBSERVED`):** centering moved the random-wave control's diversity by
`+0.04999` against my pre-registered `≤ +0.05` bound — i.e. **on the boundary**, so
I recorded the weakness rather than claiming content-specificity.

**Literature (arXiv 2608.24335, *SteerCheck*), abstract `OBSERVED`:**
> Exact replay of 960 Qwen3-14B interventions reveals complementary limits of common
> controls: isotropic directions occupy a narrow near-orthogonal region, whereas
> **sign-randomized same-construction directions often retain substantial target
> alignment**. Effect is strongly associated with signed cosine within the
> sign-randomized family (ρ=.94).

`INFERRED`: this is precisely the hazard my pre-registered bound caught. My control
**is** a random direction; the paper reports that random-direction controls can retain
real alignment. My decision to report the `+0.04999` weakness instead of asserting a
content-specific effect is supported by an external audit of the same control family.
**It does not validate the fix** — it warns that a random-direction control is weaker
evidence than it looks, which is what I recorded.

### 2.3 Harness defects move the score more than the model does

**Measured (`OBSERVED`):** S1 `pass@1` went `0/2 → 1/2` with **no change to the
generator**. The entire delta came from harness repairs: G2 was unsatisfiable
(faithful candidates tripped 50/50), a `range(idx)` off-by-one discarded the code
under test, and a `384`-token cap truncated an emission (`tokens=384 == cap`).

**Literature (arXiv 2606.17799, *Position: Coding Benchmarks Are Misaligned with
Agentic SE*), abstract `OBSERVED`:**
> …they collapse model, harness, and environment into a single end-to-end score,
> typically computed against one reference solution, with **no component-level signal
> for iteration**. … a coding agent in practice is not a model: it is a **system
> harness** — a composite of models, harnesses, contexts, environments, and feedback
> signals, **any one of which can move the benchmark score by margins comparable to
> those** [of the model].

`DERIVED`: the paper's position is the session's measured result. My three harness
defects each moved the score by more than the model did, and the reported number was
uninterpretable until they were separated. This is external support for the ordering
**repair the instrument → prove the repair is result-invariant → then measure**.

### 2.4 Where a local readout can be blind to global structure

**Literature (arXiv 2609.02141), abstract `OBSERVED`:**
> A departure of a mixed quantum state from a local Gibbs description is generally
> **invisible to local observables** but can be detected by the conditional mutual
> information (CMI).

`INFERRED`: bears on `A3` — the observational readout is wired as a **diagnostic,
never a selector** (`sagnac_mcts_planner.py:679`, `best_readout_node` tracked
separately). A local readout missing global structure is consistent with that
design choice, but the paper is about quantum mixed states and is **not** evidence
about HENRI. `HYPOTHESIS` only.

### 2.5 The contrast statistic's quantifier space

**Literature (arXiv 2608.23456, *Mutual information-entropy plane*), abstract
`OBSERVED`:** normalized permutation entropy × normalized permutation mutual
information, with a third quantity equivalent to **conditional entropy** introducing
"informational independence".

`INFERRED`: my contingency verdict reports `mi_action`, `mi_step`, and
`mi_action_given_step` with a permutation null — a conditional-MI formulation. The
paper supplies a two-axis quantifier space over the same quantities; it does **not**
license any HENRI threshold. No threshold was calibrated from it.

---

## 3. Representation-adjacent candidates (`HYPOTHESIS` — not adopted)

| ID | Title | Why flagged | Not acted on because |
|---|---|---|---|
| 2604.22863 | A wave-geometric duality for HD computing | unitary embedding bipolar HDC → coherent broadband waves; bundling = linear superposition, permutation = coherent phase, binding = nonlinear spectral mixing + engineered aliasing | It states a **unitary** embedding. HENRI's measured contract is that spatial-domain circular convolution **does not preserve norm**. Adopting the paper's claim without reconciling that contract is exactly the "unitary overclaim" my filter rejects. |
| 2607.08312 | Write-protected discrete bottlenecks | language gradients into a Gumbel-softmax bottleneck force a structural trade-off (vanilla estimator collapses to 2.2/64 symbols) | Directly relevant to egress/rank collapse and to frozen Zone C priors, but it concerns VLM→robot symbol bottlenecks, not wave→text egress. Would need its own pre-registered kill test. |
| 2609.17817 | Contaminating self-modifying coding agents with poisoned benchmarks | poisoned self-evaluation induces vulnerable code on clean held-out tasks | Bears on the **G2** contamination guard. My G2 change is a *coverage differential* target, not a self-improvement loop, so the attack surface differs. Flagged for the next contamination review. |
| 2609.09902 | Contrastive Projection | see §2.1 | **Corroboration only** — no HENRI code change is justified by a title/abstract match. |

---

## 4. What this brief does **not** establish

- **No capability claim.** These are retrieval receipts plus abstracts. A citation
  proves *what was fetched*. `routing.assert_evidence_supports_scope(..., "capability")`
  refuses a capability promotion built on retrieval-only evidence, and that refusal
  stands.
- **No AAII v4.3 claim.** 75 % of the index weight is externally graded.
- **No philosophy grounding.** That half is `BLOCKED` on auth (§0).
- **No threshold was calibrated** from any paper.
- The 12 `citation` receipts are **retrieval-only** by their own text.

## 5. Next falsification

1. Re-run both philosophy consults **after** `nlm login`, then reconcile corpus claims
   against §0's quoting of live code. If the corpus contradicts the code, the code wins
   and the conflict is recorded.
2. §2.1's convergence is testable: the paper's differencing cancels a *matched-prompt*
   shared component. My memocentering removes a *batch* mean. Whether these are the same
   operation is `HYPOTHESIS` — a named negative control (single-prompt differencing vs
   batch-mean subtraction) would separate them.
3. §2.2 requires an action, not a citation: replace the random-wave control with a
   **sign-randomized same-construction** control, which the paper reports retains
   alignment — a strictly harder control for the egress claim.
