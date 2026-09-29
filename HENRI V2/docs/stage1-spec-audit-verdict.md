# Stage-1 Spec Audit Verdict — HENRI-ARCH-2026-MVFM-Integration

**Date of measurement:** 2026-09-29
**Target:** NVIDIA RTX 5090 (sm_120, 33.67 GB), `torch 2.12.0+cu130`
**Method:** every claim below was probed from disk or executed on hardware.
Nothing in this document is adopted from the specification's prose.

---

## 1. What was executed

| Item | Result |
|---|---|
| Entrypoint named by the spec | `HENRI V2/experiments/verification/smoke_unified_vla_cuda.py` |
| On the 5090 | `UNIFIED_VLA_CUDA_SMOKE_PASS` |
| Exit gate clauses passed | **2 of 3** |

Raw smoke evidence (read from `/tmp/stage1_smoke.log` on the instance):

```
cuda available: True  NVIDIA GeForce RTX 5090
[HENRIUnifiedEgressTransducer] Loaded validated decoder checkpoint
  sha256=75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060
perceive shape: (8192, 8)  digest: 279b4609d5ff
egress checkpoint: LOADED  policy: required  trained_decoder_active: True
egress fail-closed guard: OK (generic marker refused)
code-path egress fail-closed: OK (typed guard)  out-of-vocab token id 10332
diagnostic unbinder forward: logits (1, 32000)  top_token_id 9237
act action: ACTION1  rejection: None  efe: -1.0771  explored: True
UNIFIED_VLA_CUDA_SMOKE_PASS
```

---

## 2. Exit gate, clause by clause (spec §4.1)

| Clause | Spec requirement | Measured | Verdict |
|---|---|---|---|
| (a) passage | 100% deterministic passage | `UNIFIED_VLA_CUDA_SMOKE_PASS` | **PASS** |
| (b) unitary norm | `‖Ψ‖₂ = 1.0 ± 1e-5` | `norm 1.0`, finite, dtype float32 | **PASS** |
| (c) step latency | `≤ 15 µs` on GPU | `498.37 µs` at 4×4 | **FAIL — 33× over** |

### 2.1 Clause (c) is unmet, but not for the stated reason

Clause (c) is not GPU-arithmetic-bound. Decomposition with `torch.profiler`
summed kernel durations:

| grid | wall µs | kernel µs | gap % | kernel launches/call |
|---|---|---|---|---|
| 4×4 | 498.37 | 103.75 | 79.2 | 402.1 |
| 16×16 | 123 321.25 | 3 007.11 | 97.6 | 10 100.4 |
| 30×30 | 133 697.05 | 10 588.37 | 92.1 | 35 321.1 |

**402 kernel launches for a 16-cell grid.** The dominant cost is Python-side
per-cell dispatch, not GPU arithmetic. The gap is therefore *reducible* by
vectorisation or launch coalescing — it does not require new physics.

### 2.2 Instrument defect, retracted

The first latency probe (`stage1_latency_probe.py`) classified every row
`GPU-BOUND`. It derived GPU time from `cuda.Event` timing recorded **around the
whole loop**, so elapsed-time events included idle gaps and `gpu_us ≈ wall_us`
held by construction. That verdict column carried no information and is
retracted. The replacement uses `torch.profiler` kernel sums
(`stage1_latency_corrected.py`).

---

## 3. Spec claims checked against code

| Spec claim | Status | Evidence |
|---|---|---|
| All 15 named files exist | **CONFIRMED** | all present, real byte sizes |
| `T* = 0.038316` | **CONFIRMED** | `henri_phaselock_attention.py:57` (`DOC_TEMPERATURE`); `henri_probe_calibration.py:524` (`FITTED_TEMPERATURE_60`, argmin NLL over 60 ARC tasks) |
| `τ_veto = 0.35` | **CONFIRMED value, WRONG name** | `arc_sagnac_veto.py:38` defines `DEFAULT_EPSILON_HARD = 0.35`. Independent project record: `0.35` is the search veto, `0.0431` is the pre-ZoneC setpoint. The spec must not merge them. |
| §3.2 spine = `o_vsa_ingress_tokenizer` → `arc_sagnac_veto` → `hopfield_cleanup` → `henri_calibrated_action_head` | **FALSIFIED for the named entrypoint** | the smoke imports `henri_vision_encoder`, `darwinian_phase_swarm`, `henri_action_gate`, `henri_decoder`. The §3.2 modules are reachable from `production_arc_run.py`, **not** from the smoke. |
| "unitary norm strictly checked `|‖Ψ‖−1| < 1e-5`" in `arc_sagnac_veto.py` | **NOT PRESENT** | grep for norm/unitary checks in that file returns empty |
| `D = 2048` local unit tests | **NOT USED by the entrypoint** | the smoke constructs `d_model=65536, k_blocks=8192`; `2048` appears only in older `henri_vla_engine.py` / `henri_vla_tokenizer.py` configs |
| "verified entrypoint" passes | **CONFIRMED** | see §1 |

### 3.1 Two-entrypoint finding (the spec conflates them)

Transitive local-import closure from static AST:

| entrypoint | reachable local modules |
|---|---|
| `production_arc_run.py` | **81** |
| `experiments/verification/smoke_unified_vla_cuda.py` | **32** |
| `unified_henri_vla_engine.py` | 6 |
| `henri_vla_engine.py` | 3 |

| spine module | `production_arc_run` | smoke entrypoint |
|---|---|---|
| `o_vsa_ingress_tokenizer` | YES | — |
| `arc_sagnac_veto` | YES | — |
| `sagnac_mcts_planner` | YES | — |
| `zone_c_retrieval_bridge` | YES | — |
| `arc_spatial_basis` | YES | — |
| `recursive_dual_edmd` | YES | — |
| `hopfield_cleanup` | YES | YES |
| `efe_planner` | YES | YES |
| `arc_task_functor` | YES | YES |
| `henri_calibrated_action_head` | — | — |
| `wave_jepa` | — | — |

`henri_calibrated_action_head.py` (§3.2's egress stage) and `wave_jepa.py`
(§4.2's world model) are reached by **no** entrypoint.

---

## 4. Structural audit: the "hundreds of disconnected modules"

Static AST over `HENRI V2/*.py` (imports parsed, not executed):

| metric | value |
|---|---|
| top-level `.py` modules | 197 |
| imported by ≥1 other module | 108 |
| **imported by nobody (orphan)** | **89 (45%)** |
| orphan bytes | 1 668 285 |
| orphan, capability-shaped | 27 |
| orphan, experiment-shaped | 14 |

Capability-shaped orphans include `henri_continuous_action_head.py`,
`henri_adaptive_backbone.py`, `henri_continual_learning.py`,
`henri_koopman_leaf.py`, `henri_functional_pipeline.py`,
`arc_tripartite_resonator.py`, `henri_wave_transducer.py`.

**Seven of these orphans were written in the current work session**, which is
itself the finding: new capability can be added and measured while remaining
invisible to every consumer. Wiring is the scarce resource, not module count.

---

## 5. Corrections the spec requires before Tier-2 work

1. **§3.2 spine diagram** — either point the diagram at `production_arc_run.py`,
   or add the §3.2 modules to the smoke harness. As written, following the
   diagram wires the wrong modules.
2. **Clause (c)** — either state the `≤ 15 µs` gate against a defined operation
   ("one `perceive()` call") as measured here, or reconcile it with the same
   section's `20 kHz` (50 µs period) Tier-1 cadence. Both cannot hold for the
   current implementation: the measured 4×4 path runs at ~2.0 kHz.
3. **Do not treat `0.35` as `τ_veto`.** The code name is `epsilon_hard`.
4. **Tier order is inverted by measurement.** The spec places the kernel
   substrate at Tier 0 (assumed present) and the reflex loop at Tier 1 (to be
   locked). The reflex loop's failure is *entirely* a substrate problem —
   per-cell launch count — so the Tier-0 work is a prerequisite for the Tier-1
   gate, not a later optimisation.

---

## 6. Answer to the strategic question

**Minimal viable loop first, then accrete behind contract gates.** The evidence
for that answer is in this session, not in the spec's prose:

- A smoke gate aborted a run on a closure bug **before** GPU budget was spent.
- A permutation null invalidated three prior conclusions and forced a readout
  rebuild.
- A matched three-arm control (frozen / trainable / frozen-VLM) isolated a
  mechanism that five successive hypotheses had failed to identify.

Monolithic assembly is what produced 89 orphaned modules. Gates are what make
modules safe to add.

**Amendment the spec does not make:** "minimal viable" must be defined by a
*measured* performance contract, and the contract currently fails at its own
Tier 0. Locking a reflex loop that runs at 2.0 kHz instead of 20 kHz would
freeze a known-defective baseplate and make every later tier inherit the defect.

**The repo is not a scrap heap.** It already contains a coherent measured stack
that should be *absorbed* into the accretion, not bypassed:

| capability | module | measured |
|---|---|---|
| pixels → ψ | `henri_vla_vision_ingress.py` | ψ 65 536-dim, unit norm |
| continuous control | `henri_continuous_action_head.py` | flow matching, 7-DoF |
| trainable backbone | `henri_adaptive_backbone.py` | 4.4B loads + trains, 11.4 GB peak |
| continual learning | `henri_continual_learning.py` | replay cuts forgetting +0.33 → −0.06 |
| calibrated readout | `henri_readout.py` | permutation-null validated |

---

## 7. Evidence paths

```
HENRI V2/experiments/verification/stage1_smoke_unified_vla.log
HENRI V2/experiments/verification/stage1_latency_probe_v1.py
HENRI V2/experiments/verification/stage1_latency_probe_v1_receipt.json
HENRI V2/experiments/verification/stage1_latency_corrected.py
HENRI V2/experiments/verification/stage1_latency_receipt.json
HENRI V2/experiments/verification/closure.py
HENRI V2/experiments/verification/wiring.py
HENRI V2/experiments/verification/two_entry.py
```

**No score claim.** This document records composition, entrypoint and latency
measurements only. It grants no benchmark eligibility.
