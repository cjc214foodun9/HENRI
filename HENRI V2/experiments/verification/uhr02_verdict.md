# UHR-02 VERDICT — the comparison domain, the pre-registered falsification, and the Zone C DAG

Blueprint: `HENRI-SPEC-2026-CAUSAL-REALITY-V1` (9 pages, 13,893 chars, sha256
`585e4fcdb2847c5494bfb6d3cd464778ef3f957e13910b6a88ac77a0c1e1c85f`)
Carrier: `carrier/uhr-01-homologous-representation` @ `26ef7f0` (base `adcc24e`)
**Promotion: NOT PROMOTED — `origin/main` stays at `adcc24e`.**

---

## 1. The maxim, made measurable

> "Fix the comparison's domain, not the comparison's threshold."

Two comparison domains exist for the same candidate option `U_c` and reference `R`:

```
FORM A  candidate vs the STATE it started from          <- the live loop's domain
            cos = (1/K) sum_k R_k^T A R_k
        isotropic R_k  ->  Tr(A)/8 = (|Tr U_c|^2 - 1)/8
        =>  delta_A = (9 - |Tr U_c|^2)/16               OPTION MAGNITUDE ONLY

FORM B  candidate vs the RECORDED TRANSITION            <- blueprint section 3.2
            cos ~ Tr(A_c^T A_t)/8 = (|Tr(U_c^dag U_t)|^2 - 1)/8
        =>  reads the RELATIVE group element: does the candidate PREDICT
            what the environment actually did
```

`Tr(Ad_U) = |Tr U|^2 - 1` was verified to 1e-15. FORM A's collapse was then measured
against an analytic sampling band, not asserted:

| test | result |
|---|---|
| identity `(9-|TrU|^2)/16` vs measured, K=8192 | worst err `2.950e-03` vs band `2.471e-03` (ratio **1.19**) |

## 2. The control that isolates CONTENT

Binary-searching a scale to match `|Tr U|` is **invalid** — that quantity is not monotonic
in the scale for a multi-generator product, and my first draft produced a confounded
"separation" that way (`|Tr U|` 1.9349 vs 1.1499). The correct control is **conjugation**:

```
U_B = V^dag U_A V        =>  Tr(U_B) = Tr(U_A) identically
```

| control | measured |
|---|---|
| `max |Tr U_A,k − Tr U_B,k|` (channel-wise, K=2048) | `9.537e-07` |
| `||Ad(A) − Ad(B)||_F` | `3.045` — genuinely different rotation |

## 3. Domain verdict (OBSERVED)

Separation of an EXACT equal-`|Tr U|` pair, against the 4-sigma gate:

| baseplate | FORM A | FORM B |
|---|---|---|
| ISOTROPIC | `1.348e-04` → **NO-SEP** | `3.638e-01` → SEP |
| STRUCTURED c=6 | `1.565e-01` → SEP | `4.657e-01` → SEP |

**FORM A is magnitude-blind under an isotropic baseplate.** The live loop's reference
(`boundary_batch[0]`, the per-frame residual boundary) behaves isotropically, so the live
gate cannot read option content — whatever its threshold.

## 4. The pre-registered falsification — ANSWERED without a GPU hour

Population path is **production code**: `ActionOutcomeGeneratorStore.update_generator`,
the exact function `production_arc_run.py:3001` calls.

| check | result |
|---|---|
| `theta_a[3]` norm, before → after (8 observed transitions) | `0.000000e+00` → `9.902190e+01` |
| receipt | `hermiticity_residual 0.0`, `trace_residual 3e-08`, `projection_recon_error 0.1548` |
| `STORE_POPULATED` | `True` |

| prediction | verdict |
|---|---|
| **P1** Δ leaves 0.0 | **TRUE** — trivially, since θ≠0 ⇒ U≠I |
| **P2** FORM A separates, isotropic | **FALSE** — sep `5.172e-03` < gate `1.976e-02` |
| P2 FORM A separates, structured | TRUE — sep `2.102e-01` |
| **P2 FORM B separates, isotropic** | **TRUE** — sep `3.968e-01` |
| P2 FORM B separates, structured | TRUE — sep `5.662e-01` |
| τ=0.35 in the FORM B band | **TRUE** — compliant `0.000000`, invalid `0.649520` |

**Conclusion.** Populating the store moves Δ off 0.0, satisfying the first clause — and that
clause alone proves nothing. The second clause fails on FORM A **however populated the store
is**, because the collapse is a property of the *domain*. So the honest reading is:

> **More store cannot rescue a magnitude-only comparison.**
> UHR-01's live value is not falsified for the reason the kill criterion anticipated
> (Δ *did* leave 0.0); it is superseded because the domain, not the population, was
> the binding constraint. FORM B satisfies both clauses.

τ=0.35 becomes **valid precisely when the domain is fixed** — measured proof of the maxim.

## 5. Zone C causal engram DAG (blueprint section 3) — implemented, all controls pass

`zone_c_causal_engram_dag.py` + `experiments/verification/uhr02_zonec_dag_smoke.py`:
`RESULT=ALL_CONTROLS_PASS`, EXIT 0.

| control | observed |
|---|---|
| exteroceptive gate | `ext_delta=0` → `SOLIPSISM_VETO`, nothing written |
| temporal priority | `t_dst<=t_src` → `NO_TEMPORAL_PRIORITY` |
| statistical conjunction | wrong option → `CONJUNCTION_FAILED` (residual `0.516976` > τ `0.35`) |
| edge forged | true option → `delta_pred 0.000693` |
| **dead-input negative control** | un-reinforced hypothesis **pruned**; reinforced one **survives** the same 320-tick clock; reason logged `LANDAUER_APOPTOSIS` |
| domain violation | complex operand → **RAISES** |
| attribution violation | scoring vs an unrecorded transition → **RAISES** |
| read path | gate separates true `0.000693` vs other `0.516976`, vetoed `False`/`True` |

Landauer retention: `2^(-idle/tau_retain)`, half-life form; `idle=0 → 1.000000`,
`64 → 0.500000`, `276 → 0.050328`, `320 → 0.031250`. The `kT ln 2` → joules mapping is
labelled `DERIVED` and reported as information units, never as measured heat.

## 6. Ontology grounding (NotebookLM bank `ca4bb787`, INFERRED)

Auth was expired; re-authenticated this session (`46 cookies`). The corpus **independently**
records the same mechanisms, with citations:

- *"active Engram DAG where every discovery is an immutable node containing: Parent-child
  causal edges. Typed invariant contracts. Thermodynamic utility counters. Automated
  Landauer metabolic pruning (dropping non-viable nodes when utility drops below dissipation cost)."*
- *"Boundary Axiom Circularity / The Solipsism Trap (**Falsified**): r = Ψ_state − Ψ_pred …
  the system was rewarded for being consistently wrong in the same direction … resolved by
  Objective Realignment: grounding … in exteroceptive scorecard progress (Δν)."*
- *"any branch unrolling a matching broken AST node triggers an immediate Sagnac Homodyne
  Veto (Δ_Sagnac > 0.35)."*

Corpus answers are `INFERRED`; they corroborate, and do not substitute for, the measurements above.

## 7. Tests

| suite | result |
|---|---|
| `tests/contract/test_uhr02_exteroceptive_gate.py` | **20 passed** |
| `tests/contract/test_uhr01_rfss_homology.py` | **22 passed** (regression) |
| combined | **42 passed**, EXIT 0 |

## 8. NOT DELIVERED (stated plainly)

- **Section 4** (Zone A micro-swarm → RFSS on `S^{D-1}`, D=65,536) — **not implemented.**
  It lands on the **third representation family** boundary: a complex flat `[D]` family is
  a default-OFF diagnostic sidecar with a one-way norm-preserving adapter and **no
  action-policy influence**. Implementing it as a second comparison domain would
  re-create the defect this session repaired.
- **Section 5** (nested Loops 0/1/2) — **not implemented.** Loop 0 must preserve hard
  vetoes; a habit bypass that skips the gate is a fail-open defect. Loop 1's τ is
  calibrated in FORM B only. The blueprint's `τ_0 ~ 1e-5 s` is nine orders below the
  measured `~17.8 s/step` and is recorded `HYPOTHESIS`, not built on.
- **Section 6** (Langevin substrate) — **not implemented**; existing SGLD machinery
  (`sqrt(2·T·dt)`, retraction basin) is the correct reuse point.

## 9. Next falsification (specific)

1. **Measure which inner guard stops the live θ update.** `phase820_update_info` was
   `null` at all 8 steps with no failure line, so one of `_aid >= 0`,
   `not learning_frozen()`, `su3_field is not None` failed. Probe each on the live path.
2. **Adopt the exteroceptive domain** (FORM B) in the live loop behind
   `HENRI_UHR02_EXTERO_GATE`, then re-run the paired A/B: predict FORM B separates
   compliant from invalid options against the recorded transition.
3. **Kill criterion:** if FORM B fails to separate on two consecutive live runs with a
   populated store and real role structure, the channel is magnitude-only in every domain
   and UHR-02's live value is `FALSIFIED` — the fix would then be a **content-bearing
   baseplate** (real role-filler assignments in the reference), not a threshold.
