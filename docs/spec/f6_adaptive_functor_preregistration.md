# Carrier F6: Per-Task Adaptive Functor Compilation — Pre-Registration

**Document ID:** HENRI-SPEC-2026-08-F6-ADAPTIVE-FUNCTOR
**Branch:** carrier/f6-adaptive-functor
**Status:** PRE-REGISTERED (sealed; implementation REQUIRES_APPROVAL)
**Parent verdict (ratified):** `F5_GATES_VERDICT=FALSIFIED_AT_SCALE` — governance event `5a04c478eebb…`, ledger depth 1013 (chain intact, head `10d7e03c…`, verified 2026-08-30)
**Source directive:** `Project_HENRI_F5_Post-Mortem_Audit___Per-Task_Adaptive_Functor_Directive.md` — SHA-256 `73299bd4659ed530706d52b7fa6024a70369532b9a9b324549fa1997cc5376bd`, 18,049 B, ID `HENRI-DIR-2026-08-F5-POSTMORTEM-ADAPTIVE-FUNCTOR`
**Base commit:** `7565394` (carrier/f5-structured-codec, clean worktree; F5 sealed split `658aba35…`, consumed)
**Date:** 2026-08-30

---

## 0. Executive determination

The F5 post-mortem directive is RATIFIED with two mandatory disclosed reconciliations:

- **RATIFIED:** `F5_GATES_VERDICT=FALSIFIED_AT_SCALE`. Representation geometry is solved (FPB homomorphism 1.0000, Spearman ρ 0.9377, unbinding cos 0.8781 vs F4's 0.00113); global codec approaches are terminated (arms A 0.4352 / B 0.4390 / C 0.4414 cluster; margin A−C = −0.0062 < +0.05; per-env P@1 ∈ [0.00, 0.35] demonstrates per-task phase occlusion). GO granted for Carrier F6: per-task adaptive functor compilation W_task^(e).
- **RECONCILED (harness audit, §2):** the F5 gates harness ALREADY compiled W_task per-env from each env's own 20-row demo prefix (`f5_codec_gates.py` `run_arm`, `compile_w_task`). The directive's "global static operator" framing is therefore ratified as an **operator-structure** claim, not a harness-scope claim: the F5 per-env operator was a unit-norm FHRR sum — non-unitary, no de-occlusion masking, direct cosine scoring. F6's material deltas are exactly the directive's three tiers: (a) Newton–Schulz unitary retraction, (b) subspace de-occlusion masking, (c) calibrated Hopfield lexical snapping. F6 is NOT a re-proposal of F5; it is the directive's named next step.
- **DISCLOSED CORRECTION (§3.1/3.2):** the directive's dense `K_raw ∈ ℂ^{D×D}` Newton–Schulz iteration is infeasible at D=65,536 (fp32 dense = 17.2 GB; W W† W = 5.6e14 FLOP/step). `K_raw = (1/M) Σ Ψ_Y,i ⊛ Ψ_X,i†` is a **circulant** operator (FHRR binding = circular correlation in the time domain), diagonalized by the DFT. The identical NS iteration runs as D independent complex-scalar updates `w_b ← 1.5·w_b − 0.5·w_b·|w_b|²` in the Fourier domain — same operator, same fixed point (unit-modulus), same Gate G1 (‖W†W − I‖_F computed spectrally as √(Σ_b (|w_b|²−1)²)). Dense-vs-spectral equivalence is contract test C8 at toy dimension.

---

## 1. Falsified mechanisms (do not re-propose without new evidence)

### 1.1 Static linear codebooks / egress (F2/F3/F4)
F2 `FALSIFIED_G1_KILLED`; F3 `K1_KILLED` (G1 0.178); F4 `K1_KILLED` (G1 0.2296). Egress-layer candidates closed; `HENRI_F4_EGRESS=0` standing.

### 1.2 Random-ring text ingress (Run20) and Run21 structured codec
- Run20 (commit `40ab0b0`): `encode_text` random-ring pathology; W_task = random-delta superposition.
- Run21 (commit `440f11d`, `FALSIFIED_AT_SCALE`): quantized collinear position ring `round(p_i·q_P) mod 256` — single ring orbit, integer arithmetic, no sub-bin translation, no metric anchor. F5 replaced it with exact Fourier-domain FPB (sealed). F6 inherits both dispositions; it re-proposes neither.

### 1.3 Global / pooled operator compilation
F5 measured: three codecs, per-env demo-prefix W_task, cluster at 0.435–0.441 with per-env P@1 ∈ [0.00, 0.35]. Pooling demo pairs across tasks into one operator, or compiling one codec-level operator, is FALSIFIED by that clustering. F6 compiles a per-task operator from each task's OWN demo prefix only, then retracts it to the unitary group and de-occludes.

---

## 2. F6 mechanism (four tiers, directive §3)

Inputs per task e: demo prefix rows `(X_i, Y_i)_{i=1..M}` (M = 20, bank convention), query rows X_test. All waves unit-modulus complex in ℂ^D (D = 65,536); ring quantization is a consumer-boundary convenience only (F5 rule: scoring is wave-domain).

### Tier 1 — FPB ingress (unchanged, sealed F5)
`FPBStructuredCodec` (fpb_qfhrr_codec.py) encodes action names to unit-modulus waves. Bank X rows: `psi_to_ring` → `ring_to_wave` (F5 convention). Y waves: action waves of the demo row's action.

### Tier 2 — In-situ Procrustes functor synthesis with Newton–Schulz retraction
```text
K_raw = (1/M) Σ_i Ψ_Y,i ⊛ Ψ_X,i†            (FHRR binding sum, circulant)
W     = NS(K_raw),  W_0 = K_raw/‖K_raw‖,
        w_b ← 1.5·w_b − 0.5·w_b·|w_b|² per DFT bin, 5 iterations   (directive eq. 2, spectral form)
G1: ‖W†W − I‖_F ≤ 1e-5
```

### Tier 3 — Task subspace de-occlusion masking (directive eq. 3)
```text
m_active = I( Var_demo(Arg(Ψ_X)) > ε_floor )     ε_floor = 1e-3 (pre-registered)
Ψ_goal   = normalize( m_active ⊙ (W ⊛ Ψ_X_query) )
```
Zero-variance frequency bins across the demo set are uninformative background modes; they are shunted. ε_floor is dimension-normalized phase variance.

### Tier 4 — Calibrated Hopfield lexical snapping (directive eq. 4)
```text
a* = argmax_k Re( ⟨Ψ_goal, M_k^(e)⟩ ),   M_k^(e) = action prototype waves of task e
```
M_k^(e) = mean of demo Y waves per action (per-task memory bank). P@1 vs `actions_onehot`.

**Live-code integration:** F6 is an additive, default-OFF extension of `arc_task_functor.py` (directive §5 execution order item 2): new pure kernels `spectral_newton_schulz` / `deocclusion_mask`, new `compile_adaptive_functor` + `AdaptiveFunctorCompiler` behind `HENRI_F6_FUNCTOR=1`. The existing `compile_task_functor` default path is byte-identical (Gate G6). Zero edits to `production_arc_run.py`; zero edits to the sealed F5 codec/kernels.

---

## 3. Pre-registered gates (directive §4 transcribed verbatim + F5-precedent additions)

| Gate | Metric | Threshold | Kill |
|---|---|---|---|
| G1 | In-situ functor unitary convergence: ‖W†W − I‖_F | ≤ 1.0e-5 | KILL K1 (Functor Non-Unitary Divergence) |
| G2 | Demonstration reconstruction fidelity: (1/M) Σ cos(W ⊛ Ψ_X,i, Ψ_Y,i) | ≥ 0.90 | KILL K2 (Functor Representation Deficit) |
| G3 | Grouped held-out task-level accuracy: macro P@1, 12-env broad bank, FRESH f6 split | ≥ 0.7500 | KILL K3 (Subspace De-Occlusion Failure) |
| G4 | Statistical margin over global codecs: P@1_F6 − P@1_F5 | ≥ +0.3000 | KILL K4 (Zero Adaptive Benefit) |
| G5 | Per-task phase-occlusion diagnostic: per-env P@1 across ALL 12 envs (F5 G5 precedent) | diagnostic, no threshold | — |
| G6 | Default-OFF differential: with `HENRI_F6_FUNCTOR` unset, `compile_task_functor` byte-identical (w_task_sha256 + status + cos) | exact | — |

G4 reference value: F5 sealed arm A macro P@1 = **0.4352** (event `5a04c478eebb`). Threshold ⇒ F6 macro P@1 ≥ **0.7352**.

---

## 4. Harness, split, and verdicts

- `f6_split_seal.py` — generation-only sealer, schema `f6-split-seal.v1`, seed **20260901** (NEW; must differ from consumed seeds 20260829/30/31), rule `grouped_4fold_env_disjoint_seeded_permutation_mod`, 4 folds × 3 heldout envs, `single_use=true`, receipt with npz/jsonl SHA-256, per-env counts, fold-manifest SHA-256, generator identity, UTC. Consumed-guard REFUSES f3/f4/**f5** receipt schemas.
- `f6_adaptive_functor_gates.py` — remote CUDA-only (assert `torch.cuda.is_available()`), arms:
  - **A** F6 full: FPB codec + per-task NS-retracted functor + de-occlusion mask + Hopfield snap (candidate)
  - **B** F5 control: per-env unit-norm FHRR W_task, no retraction/mask/snap (proves F6 ≠ F5)
  - **C** legacy random-ring codec (F4 baseline)
  - **D** identity / no W_task (no-supervision floor)
- Pre-registered verdicts: `F6_ADAPTIVE_PROMOTED` / `FALSIFIED_AT_SCALE` / `FALSIFIED_NO_GEOMETRY` / `BLOCKED_INFRASTRUCTURE`. Any nonzero arm exit ⇒ `BLOCKED_INFRASTRUCTURE` for the whole run (multi-arm kill matrix rule). No verdict from `--smoke` runs.
- Bank: `telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.*` (authorized capture; F5 bank, fresh split only — the F5 split is consumed).

---

## 5. Execution order (directive §5, verbatim mapping)

1. Author and stage `docs/spec/f6_adaptive_functor_preregistration.md` (this document). → SEAL `F6_PREREG_SEALED`, commit, push.
2. Implement in-situ Newton–Schulz compilation in `arc_task_functor.py` (additive, `HENRI_F6_FUNCTOR=1`, default-OFF). → contract tests RED → GREEN, SEAL `F6_IMPL_COMMITTED`, commit, push.
3. Wire de-occlusion masking to the FPB representation core (Tier 3).
4. Execute the 12-environment evaluation gauntlet on Vast CUDA (fresh sealed f6 split, arms A–D, gates G1–G6) and seal the F6 governance verdict.

## 6. Contract tests (RED first, `tests/contract/test_f6_adaptive_functor.py`)

- C1 spectral NS convergence (D=512 random wave, 5 iters, ns_err ≤ 1e-5)
- C2 dense-vs-spectral NS equivalence at toy D=64 (rel ‖W_dense − W_spectral‖_F ≤ 1e-4)
- C3 reconstruction fidelity ≥ 0.90 on synthetic Y = X ⊛ W_true demos (+ NS err ≤ 1e-5)
- C4 de-occlusion mask: constant-phase band masked, informative band kept
- C5 per-task discrimination (the occlusion claim): task A vs task B with orthogonal W_true; per-task functor margin > 0.3, pooled-global functor margin collapses ≤ 0.1
- C6 G4 arithmetic: F6 threshold = 0.4352 + 0.3000 = 0.7352
- C7 default-OFF differential: `HENRI_F6_FUNCTOR` unset ⇒ `compile_task_functor` w_task_sha256/status/cos unchanged
- C8 split-seal consumed-guard: f6 sealer refuses f5-split-seal.v1 schema
