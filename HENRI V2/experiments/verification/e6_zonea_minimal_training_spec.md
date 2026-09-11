# Carrier E6 — Zone A Minimal-Training Specification & AAII v4.3 Pipeline

**Spec:** HENRI-SPEC-2026-12-E6-ZONEA-MINIMAL-TRAINING
**Carrier:** `carrier/e6-physical-verifier` @ `10f5f23`
**Status:** SPEC (requires approval before implementation). No code written.

---

## 1. The decisive measurement

The official Artificial Analysis Intelligence Index **v4.3** was retrieved and
hashed this session (page `8ce294e6fe8c1643`, 706,687 B; receipt
`aaII_v43_pin.json` `7f0d93c4d0d6ab09`). It has **10 evaluations, weights summing
to exactly 100%**, in four categories (Agents 30 / Coding 20 / General 30 /
Scientific Reasoning 20).

Sorted by **tool access** — the property that decides whether a capability can be
supplied at test time or must live in the model:

| Tool-permitted (40%) | w | No tool (60%) | w |
|---|---|---|---|
| AA-Briefcase | 15% | AA-Omniscience | 15% |
| GDPval-AA v2 | 10% | GDP.pdf | 10% |
| Terminal-Bench v4.0 | 10% | HLE | 10% |
| AutomationBench-AA | 5% | CritPt | 10% |
| | | SciCode | 10% |
| | | AA-LCR v1.1 | 5% |

**A no-tool evaluation cannot be satisfied by retrieval, a tool loop, Zone C
engram lookup, or any external fetch.** For 60% of the index the capability must
be present in-weights or in-context.

## 2. What this does to the zero-pretraining premise

The directive states Zone A must be "highly intelligent" while being "not a
traditional ML model with world knowledge stored in static trained weights."

Those two requirements are **jointly unsatisfiable for 60% of the AAII v4.3
index.** AA-Omniscience (15%) explicitly scores factual breadth *and*
hallucination-refusal; HLE (10%) and CritPt (10%) are closed-book. A wave core
with no pretrained or ingested world model has no mechanism to answer them.

This is not a new objection. It is the same boundary already ratified: the
frozen, revision-pinned, contamination-reviewed pretrained backbone — **Contract
Amendment §2**, which entered the project with the provenance disclosure that the
"teacher" table *is* the backbone's own `embed_tokens` (event `#1374`).

**Therefore:** §2 is not merely sanctioned. It is **required** for the 60%.
The honest framing of "minimal training" is a question about the *adapter and
memory* layers, with the backbone frozen.

## 3. The three-tier minimal-training answer

### Tier 0 — zero training (targets the 40% tool block + abstention)

| Component | Mechanism | AAII target |
|---|---|---|
| Agentic tool loop | Zone A orchestrates terminal/REST/file tools | Terminal-Bench 10%, AutomationBench 5% |
| Long-context assembly | typed ingress → backbone context | GDP.pdf 10%, AA-LCR 5% |
| Calibrated abstention | fail-closed egress; refuse when unverified | AA-Omniscience non-hallucination 5% |
| Zone C engram memory | continuous accretion, no gradient | all (retrieval aids tool block) |
| File-output agent | rubric-graded deliverables | AA-Briefcase 15%, GDPval 10% |

**Trainable parameters: 0.** This tier is the whole of what HENRI can own
without touching weights, and it covers **40% of the index plus 5% abstention**.

### Tier 1 — minimal training (the only genuinely trainable part)

| Component | Shape | Purpose |
|---|---|---|
| Wave→context adapter | `[8192,8]` wave → backbone hidden space | let Zone B verification condition Zone A |
| Egress calibration head | hidden → legal response schema | task-format compliance |

- Params: order 10⁶–10⁷ (adapter scale), **not** a full fine-tune.
- Training data: **task-agnostic** alignment pairs only.
- **Benchmark-family training is prohibited**; any use of an AAII-family item in
  training voids the evaluation. Contamination review is a gate, not a step.
- Evaluation: **matched ablation** against the *unchanged* frozen backbone —
  identical prompts, decoding, evaluator, data, hardware.

### Tier 2 — rejected

Full fine-tuning / pretraining from scratch. Breaks the frozen-revision contract,
destroys the contamination boundary, and contradicts the project's own
zero-pretraining invariant. Not proposed.

## 4. Continuous learning without weight updates

The user's "continuous learning" requirement is satisfied at **Tier 0**, not by
gradients:

```
observation → Zone B wave verification (Sagnac veto, [0,2] normalized only)
            → Zone C engram write (append-only, provenance-pinned)
            → retrieval into the NEXT context
```

Zone C is the learning substrate. This is consistent with the autopoietic
directive's own energy-scarcity framing: what adapts is *which memories are
retained and retrieved*, not the weights.

## 5. Causal ordering (accepting the blueprint's inversion)

The blueprint's Directive 4 ordering is adopted **because it matches a measured
result**, not because it is asserted:

> **Decide → Superpose → Sagnac-Veto → Act**

E5b/E5c/E5d measured the opposite ordering's failure directly: generating a
bounded candidate set first and vetoing post-hoc left the gold token outside the
set, so no downstream scoring could recover it. Pre-committing the energy/priority
budget before superposition is the correction.

## 6. Gates (pre-registered, all under-specified items operationalised)

| Gate | Definition | Metric | Kill |
|---|---|---|---|
| **G-E6-0** | §2 backbone ablation is matched | identical prompts/decoding/evaluator | mismatch → BLOCKED |
| **G-E6-1** | Tier 0 scaffold beats bare backbone on the tool block | paired Δ on 40% block | Δ ≤ 0 → drop Tier 0 claim |
| **G-E6-2** | abstention is calibrated | Omniscience non-hallucination | worse than backbone → revert |
| **G-E6-3** | Tier 1 adapter is a *positive* ablation | paired Δ, bootstrap CI lb > 0 | CI includes 0 → no promotion |
| **G-E6-4** | no benchmark-family contamination | provenance scan + review | any hit → void |

**G-AUTO-1/2/3 as written are NOT runnable.** "N epochs", "variance threshold",
and "entropy must increase by X" carry no units, no estimator, and no baseline.
Each needs its own prereg with an operational definition before any remote run.
Stated numbers in the blueprint (`r = -0.427`, `p = 7.7e-05`, `α = 0.8`,
`ε = 24`, `Emax = 255`) **are verified in the paper**; the 8.4 µs kernel latency
and 12.8 µs Sagnac latency are **UNVERIFIED** — no profiler or cache telemetry.

## 7. Absent machinery (must be authored, never "wired")

Proven absent from the tree this session (363 files scanned, `_archive` excluded):

`_fused_autopoietic_kuramoto_kernel` · `qfhrr_sagnac_veto_kernel` ·
`SagnacHomodyneGate` · `ZoneBPhysicalCore` · `AutopoieticWaveMechanics`

And `henri_latent_explorer.py` (6,477 B) defines only
`is_enabled`, `_sagnac`, `_non_degenerate`, `compile_latent_goal` — it contains
**no flat parallel execution array**. The directive's premise for that file is
not supported by its contents.

## 8. Paper vs blueprint boundary (primary source, verified)

`2609.10817` (Jha, 2026-09-09) says *"a 2D spatial grid (pairings restricted to
the four immediate horizontal and vertical neighbors)"*. The string **"torus"
appears 0 times**. Paper scale is **16,384 programs / 32-byte tapes / Z80
assembly**; the blueprint's **32×32 periodic torus / 1,024 agents** is the
blueprint's extrapolation.

Implementations must cite the blueprint for torus and 1,024, the paper for grid
and the mechanistic constants. No silent conflation.

## 9. Metric defect banned by name

`Sagnacfunctor.txt`'s `max(0, 1 − Re⟨p,a⟩/D)` is **vacuous** (measured range
3e-05 at D=65,536: a dead memory passes). Corrected and certified this session to
`1 − Re⟨p,a⟩/(‖p‖‖a‖)`, range `[0,2]`, with a discriminating negative control
(seal `#f32d41b2`). **No E6 carrier may gate on the `/D` form.**

---

**Verdict:** `E6_SPEC_AWAITING_APPROVAL`. No code. No remote run. `main` = `10f5f23`.
