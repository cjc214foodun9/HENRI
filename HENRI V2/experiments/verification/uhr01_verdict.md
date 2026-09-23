# UHR-01 VERDICT — homologous representation, remote paired A/B

Carrier: `carrier/uhr-01-homologous-representation` @ `d825d98` (base `adcc24e`)
Surface: Vast.ai `52189427` — NVIDIA RTX PRO 5000 Blackwell, 48935 MiB, compute_cap 12.0
Overlay: `799,034,119 B`, sha256 `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060` (remote-verified)
Receipts: `C:\Users\chan\AppData\Local\Temp\uhr01_egress\`
Promotion: **NOT PROMOTED** — main stays at `adcc24e`.

## 1. What the defect was (OBSERVED)

`SU3FieldWaveTransducer.field_to_wave` emits a **complex flat** `[65536]` wave; the Zone C
boundary waves are **real** `[num_blocks, 8]`. The baseline gate compared the two by raw
index pairing, so the similarity was ~0 by construction:

| arm | delta_axiom (8 steps) | hard_vetoed |
|---|---|---|
| BASELINE (`HENRI_UHR01_RFSS` unset) | 0.997175, 0.997528, 0.997542, 0.998115, 0.998176, 0.99851, 0.999185, 0.999582 | **8/8 True** |

The gate was truthful — it reported that the two objects were unrelated. The defect was the
*comparison's domain*, not the threshold.

## 2. What the repair does (OBSERVED)

SU(3) acts on its own 8-dimensional Lie algebra by the **adjoint** representation, which is a
real orthogonal map `Ad: SU(3) -> SO(8)` — and the axiom block **is** an 8-vector.

| property | measured (local + CUDA) |
|---|---|
| orthogonality `‖A·Aᵀ − I‖` | `1.1e-07` |
| `det(A)` | `1.000000` (SO(8)) |
| homomorphism `‖Ad(UaUb) − Ad(Ua)Ad(Ub)‖` | `8.9e-08` |
| anti-order control | `1.2e+00` (order rejected → 4-step composition preserved) |
| character law `Tr(Ad U) = |Tr U|² − 1` | exact to 1e-15 |
| block-norm deviation | `1.192e-07` (loader tolerance `1e-4`) |

Placement: `uhr_rfss.py` implements the binding; `OPINEObjectMCTS.project_to_boundary_family`
delegates (one reader, two consumers); `production_arc_run.py` consumes it behind
`HENRI_UHR01_RFSS` (default OFF, line 279).

## 3. Remote paired A/B result (OBSERVED)

Instance `52189427`, mode `phase823_live_gauntlet --envs 1 --steps 8`, both arms exit 0.

| step | BASELINE d_axiom | BASELINE veto | RFSS d_axiom | RFSS veto | RFSS `uhr01` |
|---|---|---|---|---|---|
| 0 | 0.999185 | True | 0.0 | False | rfss_role_filler, nb=8192, ng=4, bnd=1.192e-07 |
| 1 | 0.999582 | True | 0.0 | False | same |
| 2 | 0.997542 | True | 0.0 | False | same |
| 3 | 0.998176 | True | 0.0 | False | same |
| 4 | 0.997175 | True | 0.0 | False | same |
| 5 | 0.998510 | True | 0.0 | False | same |
| 6 | 0.997528 | True | 0.0 | False | same |
| 7 | 0.998115 | True | 0.0 | False | same |

`engaged`: False (baseline) → **True** (RFSS). Steps differing: **8/8** in both channels.
The flag reached the process and the projection executed end-to-end.

## 4. Pre-registered gate scoring

| id | criterion | verdict | evidence |
|---|---|---|---|
| **C1** | synthetic bidirectionality at tau=0.35 | **PASS** | compliant Δ `0.081–0.187` (0/8 veto) vs invalid Δ `0.485–0.559` (8/8 veto); identity Δ `0.000000`; permutation control separates |
| **C2** | live-loop Δ population leaves `[0.99, 1.0]` | **PASS (VACUOUS)** | population **is** `0.0` — but see §5: it is a self-comparison ceiling, not axiom-content grounding |
| **C3** | default path byte-identical | **PASS** | baseline retained the legacy signature `0.997–0.9996`, veto 8/8; `uhr01` absent; contract suite green with flag OFF |
| solve rate | diagnostic only | **0.0 both arms** | `score_eligible=false`, `score_block_reason=LOADED_COMPONENT_NOT_ON_ACTION_PATH`, `action_head_load_status=SKIPPED_POLICY_DISABLED` |

## 5. The honest limit (DERIVED — the load-bearing finding)

Δ_axiom = **exactly 0.0** for all 8 RFSS steps is a degeneracy signature, and I traced its cause:

```text
theta_a never updated live (phase820_update_info = null at all 8 steps)
lie_element:  1j * einsum(theta, basis)   ->  zero matrix
construct_macro_option: matrix_exp(0)     ->  U = I
Adjoint action: Ad(I)                     ->  I
projection returns the axiom itself       ->  delta(cand, axiom) = 0.0
```

**CORRECTION (caught in review, 2026-09-23).** An earlier draft of this section
attributed the zeros to `preference_store_size = 0`. That was a **conflation of two
different stores**, and it is wrong:

| store | class | what it holds | telemetry field |
|---|---|---|---|
| pragmatic prior | `ContinuousHopfieldCleanup` (`efe_planner.py:323`) | waves from transitions with valence > 0 | `preference_store_size` |
| action outcome generator | `ActionOutcomeGeneratorStore` (`henri_external_outcome_refactor_module.py:34`) | per-action su(3) angles `theta_a` | `phase820_update_info` |

`lie_element` reads `self.theta_a`, i.e. the **generator** store. `preference_store_size`
counts the **Hopfield** store. Both were zero/false in the run, but one is not the cause
of the other. The measured cause is that `phase820_update_info` was `null` at every
step, so `theta_a` was never written and stayed at its `torch.zeros(...)` initial value.

The update call at `production_arc_run.py:3001` is real and reachable — it is guarded by
`HENRI_ARC_ACTION_EFE and action_outcome_store is not None and obs_next is not None and
obs_next.frame`, and mode `phase823_live_gauntlet` force-sets `HENRI_ARC_ACTION_EFE=1`
(line 497). No `[phase820] update failed` line was emitted, so the inner guard
(`_aid >= 0 and not learning_frozen() and su3_field is not None`) is where it stopped.
Which of those three failed is **not yet measured** — it is the next probe, and it is
distinct from the domain question below.

The store-population probe (`uhr02_store_population_probe.py`) shows the production
`update_generator` path works when driven: `theta_a[3]` norm `0.000000e+00` ->
`9.902190e+01` over 8 observed transitions, receipt `hermiticity_residual 0.0`,
`trace_residual 3e-08`, `projection_recon_error 0.1548`.

Zero-generator control (local, measured): `delta = -2.28e-07` (cos `1.0000004`).
Non-zero control at `theta ≈ 0.10`: `delta = 0.35076` — i.e. **at the tau boundary**.

So the live gate flipped from *always-veto* (8/8) to *never-veto* (0/8). Both extremes are
non-discriminative: with an un-updated generator the candidate carries no option content
and **is** the axiom. UHR-01 made the veto *computable* and removed the false 8/8
rejection; it did **not** establish that the live gate discriminates candidate quality.

**And populating the store does not fix that** — see `uhr02_verdict` below: the collapse
is a property of the comparison *domain*, not of the store's population.

No algebra embedding is claimed: SU(3) is **not** representable in Cl(3,0) or Cl(1,3).

## 6. Next falsification

1. Populate the outcome store (`preference_store_size > 0`) and re-run the paired A/B; predict
   Δ_axiom moves **off** 0.0 and separates compliant from invalid options in the live loop.
2. If Δ stays 0.0 with a populated store, the generator path is not feeding the projection —
   inspect `lie_element` → `_gens` → `project_to_boundary_family` for a dead variable.
3. Kill criterion: if two further runs cannot move Δ off 0.0 with a populated store, UHR-01's
   live-loop value is `FALSIFIED` and the module stays archived, not promoted.

## 7. Cost

credit `$16.49` after the run; all six instances confirmed `exited|stopped` by re-read;
no active GPU.
