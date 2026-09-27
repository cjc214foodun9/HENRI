# HENRI — Ontologically Grounded Action Plan: The Five Structural Gaps

**Document:** `HENRI-PLAN-2026-09-27-FIVE-GAPS-V1`
**Base commit:** `0ad675e` (branch `carrier/zone-a-selfplay`)
**Method:** autoresearch `orx` receipts + own-tool premise audit of every document claim
**Standard:** ADS-STE100. Labels: `OBSERVED` `DERIVED` `INFERRED` `HYPOTHESIS` `FALSIFIED` `BLOCKED`

---

## 0. Premise audit — four document claims checked before use

Per the integration rule, no claim about repository state gates a decision until an
own-tool call confirms it. Four claims in the supplied document failed or narrowed.

| # | Document claim | Own measurement at `0ad675e` | Verdict |
|---|---|---|---|
| 1 | `arc_task_functor.py` holds `W = (X^T X)^(-1) X^T Y` | Real code is a **per-slot DIAGONAL ridge** solve: `numerator / (denom + reg_lambda)`, elementwise, reducing only over the demo axis. The module's own docstring records the **FFT/circulant family was FALSIFIED** and that this diagonal family reaches **held-out 0.4368 / in-sample 0.7645 on 60 real ARC tasks** — vs ≈0.00 for the FFT family | **FALSIFIED (literal)** |
| 2 | `arc_sagnac_veto.py` uses `cosine_similarity > 0.95` | `DEFAULT_EPSILON_HARD = 0.35`; the string `0.95` occurs **nowhere** in the file | **FALSIFIED (literal)** |
| 3 | `henri_wave_transducer.py` is missing | **ABSENT** — confirmed | **OBSERVED** |
| 4 | Re-run `execute_authentic_coding_benchmark.py --benchmark scicode --window 48` | That binary declares **0 `add_argument` calls** and mentions "scicode" **once**. The cited invocation cannot run as written | **FALSIFIED (literal)** |

**Consequence.** Gaps 2 and 3 as *worded* target the wrong code. The real defects are
narrower and different (see §2). Reporting them as written would produce a refactor
of a family that is already the measured-best one.

---

## 1. Autoresearch grounding (orx 0.2.10, no login)

Receipts recorded to `.ar/receipts.jsonl`. **The evidence rule applies: a receipt
proves WHAT WAS FETCHED, never what is true.** An empty candidate set is an empty
set, not proof of absence.

| Query | Relevant candidates returned |
|---|---|
| Q1 VSA transformation factorization | `2608.18404` Vector Symbolic Policy Gradient · `2608.02807` Learning a Vector-Symbolic Model for Socio-Cultural Tasks · `2609.01408` NeuSOGA · `2608.06614` Factorized Hypothesis Search |
| Q2 Dense associative memory / Hopfield | `2607.29554` **Exponential Capacity in Multilayer Hetero-Associative Networks** · `2606.18492` Dense Holographic Associative Memories · `2604.01469` Oscillator-Based Associative Memory with Exponential Capacity · `2604.07401` Geometric Entropy and Retrieval Phase Transitions in Continuous Thermal Dense Associative Memory · `2601.00984` Biologically Plausible Dense Associative Memory |
| Q3 Spatial topology / containment | `2605.14068` **CurveBench: Exact Topological Reasoning over Nested Jordan Curves** · `2605.16785` Encoding Robust Topological Signatures for HDC · `2607.28287` Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3 · `2609.26046` Canonical locks that encode part-whole hierarchies · `2608.14220` Generalized Parallelogram Rule for Proportional Analogies on Riemannian Manifolds · `2604.22863` A wave-geometric duality for HDC |
| Q4 Continuous prefix into a frozen LM | `2609.23601` PREM: Prefix-Steered Recurrent Memory |

**Two candidates are load-bearing for this plan:**
- **`2605.14068` CurveBench** grounds Gap 1: exact topological reasoning over nested
  Jordan curves is a *published, benchmarked* problem — so the "Jordan curve
  homology" prescription in the document is not novel and has an external yardstick.
- **`2607.29554`** grounds Gap 5: exponential capacity in **multilayer**
  hetero-associative networks — i.e. the capacity result the document attributes to
  a single-layer Hopfield snap requires a *multilayer* construction to hold.

Gap 4 (thermal recoil) returned no direct hit; it stays `HYPOTHESIS`.

---

## 2. The decisive prior art — the resonator already exists and it returned VOID

**This is the most important finding in this plan.**

`carrier/aaii-v43` @ `16d573c` already contains the tripartite resonator:

```
HENRI V2/henri_resonator.py            (TripartiteResonator, FactorCodebook,
                                        ResonatorConfig, ResonatorResult)
HENRI V2/arc_tripartite_resonator.py
HENRI V2/tests/contract/test_tripartite_resonator.py
HENRI V2/tests/contract/test_resonator_kill_gates.py
HENRI V2/experiments/verification/KILL_PREREGISTRATION_resonator_carrier.md
```

Commit message, verified: *"tripartite VSA resonator — instrument VALIDATED
(28/28, mutation gate 4/4), **VOID on real ARC**"*.

**Measured verdict table from that commit (`OBSERVED`):**

| Scenario | Branch | Treatment | Closest control |
|---|---|---|---|
| solvable | TREATMENT_STRICTLY_BEATS_ALL | 1.0000001 | shuffled +0.3672 |
| identity_truth | VOID: control at/above treatment | 1.0000000 | identity (tie) |
| impossible | VOID: cap hit + 4 controls beat | −0.00146 | diag_ls, identity |
| **real_arc** | **VOID: cap hit + 2 controls beat** | **0.26858** | **diag_ls, identity** |

**Read plainly: on the real ARC task, `identity` beats the resonator, and the
relaxation hit its iteration cap (non-convergent).** The same commit confirms the
hardcoded `0.0431` is **self-vetoing** and that *"iterating on 0.0431 is a
non-termination trap"*.

The pre-registered kill file also records the structural reason: measured on the
live encoder, **colour is NOT a per-slot diagonal op** (`|z_B/z_A|` std 37.94) and
**the mask is NOT diagonal** (`|ratio|` std 14975.8) — so "solve for the factors"
is forbidden by measurement; only **roll** is provably diagonal.

**Therefore ACTION 2 as written would re-run a measured VOID.** Deploying
`T_task = R_torus ⊗ M_mask ⊗ Spin(3)` as the *replacement* for the diagonal functor
means replacing a family with measured held-out **0.4368** on 60 real ARC tasks with
a family measured at **0.26858 < identity** on the one real task it was tried on.

---

## 3. The remedy plan, gap by gap

### Gap 1 — Sensory ingress: global spatial mean-pooling
**Status:** `HYPOTHESIS` (not yet measured by me).
**Remedy:** build `henri_topological_encoder.py` as a **default-OFF sidecar**.
- Do NOT touch `o_vsa_ingress_tokenizer.py`'s live path; that is a separate
  representation family and the architecture skill forbids silent merges.
- Gate K1a: two grids differing only in object position must separate by
  `≥ 1e-3` in `‖·‖_inf`.
- **Negative control (must FAIL):** a content-blind branch computing its phasor
  from index buffers only. If the control passes, the gate is vacuous.
- External yardstick: `2605.14068` CurveBench (nested Jordan curves).

### Gap 2 — Relational task core
**Status:** the document's premise is `FALSIFIED`. The live operator is NOT a matrix
inverse; it is the measured-best diagonal family on 60 real ARC tasks.
**Remedy (re-scoped):** do **not** excise the diagonal solve. Instead:
1. Keep it as the control arm.
2. Add the resonator as a **third, default-OFF arm** behind a named flag.
3. Pre-register the A/B: accept the resonator **iff** it strictly beats the diagonal
   arm on the **same held-out split**, `hit_rate − identity_rate ≥ 0.30`, and it
   converges below its iteration cap.
4. **Prior evidence says it will not.** The expected outcome is a re-confirmed
   `FALSIFIED`, logged in `evolution.jsonl` so it is never re-proposed blind.

### Gap 3 — Verification gate (Sagnac)
**Status:** literal claim `FALSIFIED`; the live gate is `0.35`, advisory, and its
contract is `(delta, coherence, vetoed, status)`.
**Remedy:** the **two-stage separation** is already recorded in the ontology
(`0.35` = search veto; `0.0431` = pre-Zone-C crystallization setpoint). Any new
phase-winding kernel must keep that split. Do **not** promote it to a hard selector:
the live ARC action policy is `EFEPlanner.select_action`.

### Gap 4 — Thermal recoil loop
**Status:** `HYPOTHESIS`. No direct literature hit.
**Remedy:** wire `I_dark` into the SGLD thermal kick — but it is currently
**flag-gated OFF** (`HENRI_DREAM_CREEP`). Enabling it is a load-bearing math change
and requires explicit approval. Cheapest kill: if the dream update never changes the
**selected action**, it stays a diagnostic and must never touch egress (the M3-ACT
latch already enforces this in code).

### Gap 5 — Action / code egress
**Status:** `BLOCKED`. `henri_wave_transducer.py` is `OBSERVED` absent, and the
cited re-run command cannot execute.
**Remedy:** build the transducer against `SPEC-2026-09-24-FUWT-EGRESS-V1`
(`[B,65536] complex64 → [B,32,6144] polar → [B,32,2048] prefix`, fail-closed on
`‖ψ‖ ≠ 1 ± 1e-4`). Use the **existing** `ContinuousHopfieldCleanup.lexical_snap`
(β configurable) rather than a new snap. Then:
- **Additional finding from the kill file:** `ContinuousHopfieldCleanup.store_engrams`
  **accepts rank-3 input** `[1,8,8]` without validation — a real defect. The
  transducer must flatten explicitly.
- The SciCode score claim stays `BLOCKED`: the cited harness takes no such flags,
  and prior measurement on `main` recorded that **no HENRI code path can emit a
  SciCode solution**. Gate on the FUWT landing first.

---

## 4. Pre-registered kill criteria

| Gate | Reject if |
|---|---|
| Topological encoder K1a | position separation `< 1e-3`, **or** the index-buffer control passes |
| Resonator A/B | it does not strictly beat `diag_ls` on held-out, **or** it hits the iteration cap |
| Sagnac kernel | a dead (all-zero) field does **not** fire the veto |
| Hopfield snap | rank-3 engrams are still accepted without validation |
| Any egress change | the dream update does not change the **selected action** (M3-ACT) |

A negative result is a governance win. It is recorded, not masked.

---

## 5. What this plan does NOT claim

- No benchmark score. No ARC-AGI-3, SciCode, or AAII number is asserted.
- No ICL emergence. Stage-0 is a token-accumulation job.
- No architectural equivalence between HENRI and any cited paper.
- The `orx` receipts prove retrieval only.
