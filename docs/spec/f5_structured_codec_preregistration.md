# Carrier F5: Structured Compositional qFHRR Codec — Pre-Registration

**Document ID:** HENRI-SPEC-2026-08-F5-STRUCTURED-CODEC
**Branch:** carrier/f5-structured-codec
**Status:** PRE-REGISTERED (sealed; implementation REQUIRES_APPROVAL)
**Parent verdict (ratified):** `F4_GATES_VERDICT=K1_KILLED` — governance event `392a23ff`, audit hash `494296385f768c1d…`, ledger depth 1009
**Source directive:** `Project_HENRI_F4_Post-Mortem_Audit___Structured_Codec_Pre-Registration_Directive.md` — SHA-256 `09c0fccc365e02267f9e514dc47eee56eadc795ec7304e809779ba5181e7a207`, 16,938 B, ID `HENRI-DIR-2026-08-F4-POSTMORTEM-CODEC-REFORM`
**Date:** 2026-08-30

---

## 0. Executive determination

The F4 post-mortem directive is RATIFIED with a mandatory disclosed reconciliation:

- **RATIFIED:** F4 egress layer exhausted → pivot to representation geometry. `F4_GATES_VERDICT=K1_KILLED` ratified (arm A macro P@1 0.2296, margin +0.1437; paired bootstrap lb ≤ 0 for nonlinearity, task conditioning, in-situ adaptation). Root cause identified as `encode_text` random-ring pathology (K2 unbind cos 0.00113 ≪ 0.99).
- **RECONCILED (pre-registration, directive §4):** the directive is SILENT on Run21, the prior structured-codec kill (2026-08-04, commit `440f11d`, verdict `FALSIFIED_AT_SCALE`). Per the directive-handling rule, this pre-registration embeds the full Run21 disposition (§2) so that F5 cannot be read as a silent re-proposal of a killed mechanism.
- **DISCLOSED CORRECTION (§3.1 mechanism):** the directive's ASCII diagram (§3) depicts a *coordinate harmonic* `φ_coord(x,y) = exp(i[x k_x + y k_y])` with a **3D spatial grid**; its equation block (eq. 1) prescribes **1D Fourier-domain Fractional Power Binding** (FPB). The 3D-coordinate carrier is not compatible with the current text-ingress problem (no ARC grid coordinates in the F3/F4 12-env trajectory bank) and would be a separate representation family requiring its own boundary (§: complex-wave-family-sidecar-boundary). F5 binds to the **1D FPB** of eq. 1 — the continuous-position mechanism that Run21 approximated but did not implement exactly. The 3D coordinate-harmonic variant is recorded as a future-bounded carrier, NOT part of F5.

---

## 1. Falsified mechanisms (do not re-propose without new evidence)

### 1.1 Static linear codebooks (F2/F3/F4, egress)
- `F2HopfieldEgressCodebook` (F2, `#479d8f0f` `FALSIFIED_G1_KILLED`); F3 dual-ridge `M = YᵀΨ(ΨᵀΨ + λI)⁻¹` (`F3_GATES_VERDICT=K1_KILLED`, G1 0.178); F4 MLP+pre-inversion+SGLD (`F4_GATES_VERDICT=K1_KILLED`, G1 0.2296). All egress-layer candidates closed; `HENRI_F4_EGRESS=0` standing. No egress experiment without a NEW pre-registration.

### 1.2 Random-ring text ingress (Run20, 2026-08-03, commit `40ab0b0`)
- `qFHRREpistemicCodec.encode_text` = SHA-256-seeded `torch.randint` → every distinct string maps to an independent random Z_256 ring; similarity ≈ 1/√D ≈ 0.0039 for ALL distinct pairs. A `W_task` compiled from these waves is a random-delta superposition; the task relation is not representable in the operator. FIX DIRECTION was already identified as a structured compositional codec (token engrams + position binding).

### 1.3 Run21 structured char-position codec (2026-08-04, commit `440f11d`, `FALSIFIED_AT_SCALE`)
- `StructuredCharPositionCodec` (`qfhrr_structured_codec.py`, `experiment/structured-codec` branch, verdict doc `9246c08`): char engrams + quantized fractional position `q_pos(i,d) = round(p_i · q_P[d]) mod 256` (p_i = i/max(1,n−1)), complex-phase bundling, `ring_to_real` = cos(phase) at every consumer.
- **Measured (OBSERVED, RTX 5090):** nearby_input_sim 0.62 (vs legacy −0.0045, 10× baseline gate PASS); position_swap_sim 0.0066 (full) vs 1.0 (nopos) — order sensitivity active; task 89 improved 42→7; **task 62 stayed occluded (30/52)**; joint (62,89) ≤ 24 gate unmet; no variant beat identity/legacy jointly → **`FALSIFIED_AT_SCALE`**.
- **Attribution (HYPOTHESIS, sealed):** compositional geometry improvement is NOT sufficient for task-level operator success; per-task phase occlusion (run19 pattern) persists; a single global codec does not resolve task-specific occlusion. **The skill's named next step was "per-task adaptive codec/position scale" — NOT more global codec tuning.**
- **Why F5 is NOT a silent re-proposal (the material difference):** Run21 implemented FPB only in its **quantized integer-ring limit** — `round(p_i · q_P) mod 256` is a collinear integer scaling of one base ring, which (a) is not a true fractional power (sub-bin phase translation is impossible in Z_256 integer arithmetic), (b) is collinear by construction (all positions lie on the same ring orbit — which is exactly why `position_mode="independent"` was added as a repair), and (c) has no metric anchor. **F5 implements the exact Fourier-domain FPB** `Ψ^x = ℱ⁻¹(exp(i·x·Arg(ℱ(Ψ))))` — continuous phase rotation per dimension, no integer quantization of the exponent, one base ring per bound dimension — plus the directive's metric-anchor invariant and an explicit per-dimension position basis (the run19-attributed per-task/position-scale dimension). This is a NEW mechanism within the codec class, with its own falsifiable gates.

---

## 2. F5 mechanism (three tiers + metric anchor)

Inputs: text token sequence `t_0..t_{n-1}` (char or word atoms), base phase rings per dimension.

### Tier 1 — Continuous Fractional Power Binding (FPB, Fourier domain)
For a base ring `Ψ_base ∈ S^{D-1}` (real, seeded deterministically), the fractional power at continuous position `x`:

```text
Ψ(x) = ℱ⁻¹( exp( i · x · Arg(ℱ(Ψ_base)) ) )   (directive §3.1, eq. 1)
```

with `ℱ` the real→complex DFT (torch.fft.rfft), `Arg` the per-bin phase, `x ∈ [0, n)` (not normalized — position is a continuous coordinate along the ring's geodesic). This is exact sub-bin phase translation: `Ψ(x) ⊛ Ψ(y) = Ψ(x+y)` (homomorphic), and `⟨Ψ(x), Ψ(y)⟩ = (1/D)·Σ_d cos(x·ω_d − y·ω_d) = (1/D)·Σ_d cos((x−y)·ω_d)` — a smooth function of `|x−y|`, NOT a random ring.

### Tier 2 — Category-theoretic role/type binding
Each dimension's base ring is assigned a semantic role (spatial coordinate, semantic role, primitive type) and bound via modular phase addition (Z_256 addition or continuous phase addition):

```text
Ψ_token = Ψ_char(t) ⊛ Ψ_pos(x) ⊛ Ψ_role(role)   (⊛ = elementwise phase addition)
```

Syntactic permutation changes the phase spectrum deterministically (order sensitivity).

### Tier 3 — Homomorphic metric preservation (directive §3.1, eq. 3)
```text
⟨Ψ(T1), Ψ(T2)⟩ ≥ 1.0 − κ·d(T1, T2)²
```
with `κ` a codec constant (measured, not assumed) and `d` the edit/position distance. This is the metric-anchor invariant; Gate G1 tests it directly.

**Live-code integration:** F5 is a **new default-OFF module** `fpb_qfhrr_codec.py` (per the representation-family boundary, a new codec family is a separate module, not a silent mutation of the live `qfhrr_kernels.py` similarity kernels). It exposes `encode_text`/`bind_hadamard`/`unbind_hadamard`/`compute_similarity` with the same contract as `StructuredCharPositionCodec` so downstream consumers (rank probe, F3/F4 bank harness) can select it by flag `HENRI_F5_CODEC=1`. Zero edits to `production_arc_run.py`; zero edits to the live `qfhrr_kernels.py` (default path byte-identical).

---

## 3. Pre-registered gates (acceptance / rejection)

| Gate | Metric | Threshold | Notes |
|---|---|---|---|
| G1 | **Homomorphic metric continuity:** Spearman ρ(⟨Ψ1,Ψ2⟩, −d(T1,T2)) | ≥ 0.85 | directive §4 G1; measured over a curated pair set (nearby/swap/far) |
| G1b | FPB homomorphism: ‖Ψ(x)⊛Ψ(y) − Ψ(x+y)‖ (or cos ≈ 1) | ≥ 0.99 cos | NEW — proves exact FPB, the Run21 gap |
| G2 | **Task-functor unbinding coherence:** cos(Ψ ⊛ W_task†, Ψ_action) | ≥ 0.40 | directive §4 G2 (F4 measured 0.00113) |
| G3 | **Grouped 4-fold held-out generalization:** macro P@1 | ≥ 0.8000 | directive §4 G3, 12-env broad bank, fresh sealed split (F4 split is consumed) |
| G4 | **Margin over random-hash baseline:** P@1_structured − P@1_random | ≥ +0.5000 | directive §4 G4 (F4 margin +0.1437) |
| G5 | **Per-task phase-occlusion diagnostic (NEW, the Run21 gap):** report per-task P@1 across ALL 12 envs, not just the 3 heldout | — | diagnostic; no threshold; the Run21-attributed occlusion pattern must be quantified |
| G6 | **Default-OFF differential:** with `HENRI_F5_CODEC` unset, live `qfhrr_kernels` similarity path byte-identical | exact | proves zero bleed into the default path |

**Pre-registered verdicts:** `F5_CODEC_PROMOTED` (G1+G1b+G2+G3+G4 all pass) · `FALSIFIED_AT_SCALE` (any absolute gate fails — the Run21 verdict class, indicating occlusion persists) · `FALSIFIED_NO_GEOMETRY` (G1/G1b fail — FPB not implemented correctly) · `BLOCKED_INFRASTRUCTURE` (nonzero arm exit / missing artifact) · `BLOCKED_TARGET_LEAKAGE` (provenance scan fails) · `CONDITIONAL_REUSED_EVAL` (fresh-split discipline violated — forbidden).

---

## 4. Data path and split policy (anti-leakage)

### 4.1 Corpus
Identical to F4: `trajectories_production_run_f3v2.npz` — `psi [1536,65536] f16`, `next_wave [1536,65536] f16`, `actions_onehot [1536,7] u8`, `action_names (7,)`; per-env counts as F4 §4.1. Hash pinned in `f3_split_seal.json` (`npz_sha256`).

### 4.2 Fold policy — FRESH SEAL REQUIRED (F3 AND F4 splits are consumed)
The F3 split (seed 20260829, digest `30504659…`) and F4 split (seed 20260830, rule `grouped_4fold_env_disjoint_seeded_permutation_mod`, digest `640763c6…`, single_use) are BOTH CONSUMED and will NOT be reused. F5 seals a NEW grouped 4-fold split: env-disjoint (3 heldout envs × 4 folds), NEW seed (default `20260830` — different permutation), `single_use=true`, full SHA-256 receipts, generation-only process. Any F5 result on a consumed split is `CONDITIONAL_REUSED_EVAL` and fails the carrier.

### 4.3 Demo prefix (test-time task conditioning)
Same rule as F4 §4.3: per-env rows ordered by capture time; first `k=20` rows (floor min(20, n_env/5)) form the demo prefix; remaining rows are evaluation rows. Heldout-fold envs use demo prefix ONLY for W_task compilation at eval time; NEVER in codec training (the codec is deterministic/frozen — zero trainable parameters) and NEVER in evaluation rows. Static provenance scan (pre-run) asserts zero heldout-eval-row leaks.

### 4.4 Hyperparameters (fixed, no tuning on heldout)
Seed 20260830; FPB base rings: SHA-256-seeded deterministic rings per dimension (token/char, position, role); position coordinate `x = i` (0-indexed) — NOT normalized (the FPB geodesic is translation-invariant, and normalization to [0,1] would collapse long strings); bundling = complex-phase addition then quantize to Z_256 (or continuous phase, kept in the ring domain for downstream qFHRR compatibility); `κ` in G3 = 1.0 (default, measured). No trainable parameters; no optimizer.

---

## 5. Arms (multi-arm kill matrix)

| Arm | Codec | W_task | Purpose |
|---|---|---|---|
| A | **FPB (full F5)** | compiled from FPB waves | full structured codec |
| B | Run21 quantized `StructuredCharPositionCodec` | compiled from Run21 waves | **control** — proves F5 ≠ Run21 (the killed class) |
| C | legacy random-ring `qFHRREpistemicCodec` | compiled from legacy waves | control — the F4 baseline (G4 margin reference) |
| D | identity (prompt-wave reference) | — | no-supervision baseline (Run21 convention) |

Any nonzero arm exit → `BLOCKED_INFRASTRUCTURE` for the whole run. Arm B is mandatory: if F5 ≈ Run21 on the bank, the kill is confirmed and the occlusion hypothesis (not the mechanism) is the binding constraint.

---

## 6. Cheapest kill experiments (pre-registered, run BEFORE full gates)

1. **FPB homomorphism (kill 1):** `cos(Ψ(x)⊛Ψ(y), Ψ(x+y)) ≥ 0.99` for x,y ∈ {1..8} — proves the FPB is real (G1b). Failure = the FPB is not implemented correctly → `FALSIFIED_NO_GEOMETRY`, fix harness, no verdict.
2. **Metric continuity (kill 2):** Spearman ρ(cos, −d) ≥ 0.85 on curated pairs (nearby/swap/far) — proves geometric continuity (G1). Failure = `FALSIFIED_NO_GEOMETRY`.
3. **Unbinding coherence (kill 3):** cos(Ψ ⊛ W_task†, Ψ_action) ≥ 0.40 on heldout-env rows — proves the task relation is representable (G2). Failure = `FALSIFIED_AT_SCALE`-class, no bank run.
4. **Default-OFF differential (kill 4):** with `HENRI_F5_CODEC` unset, `qfhrr_kernels` similarity output byte-identical (G6). Failure = wiring defect, fix, no verdict.
5. **Leakage scan (kill 5):** static provenance audit of every tensor entering W_task / evaluation (per §4.3). Failure → `BLOCKED_TARGET_LEAKAGE`, stop.

---

## 7. Resources and constraints

- Remote CUDA only for bank re-evaluation: Vast RTX 5090 (32 GB). FPB kernels: rfft/irfft on [1,65536] f32 per string — trivial memory (~512 KB/string), CPU-fallback pure torch (no Triton dependency).
- Local CPU tests: contract/shape only, never verdicts.
- Zero edits to `production_arc_run.py`; zero edits to live `qfhrr_kernels.py` (default path byte-identical, G6).
- New module `fpb_qfhrr_codec.py` (default-OFF flag `HENRI_F5_CODEC=1`); contract tests `test_f5_fpb_codec.py`; remote metric-preservation harness `f5_metric_preservation.py` (CUDA, standalone).
- Dense `[D,D]` tensors are prohibited anywhere (K4); FPB is elementwise + FFT only.

---

## 8. Verification plan (TDD, then remote)

1. Contract tests first (RED): FPB homomorphism, metric continuity, role binding order sensitivity, W_task compile from FPB waves, unbinding coherence, default-OFF differential, fold-loader provenance scan, gate arithmetic. GREEN 20+ tests.
2. Exact-SHA detached worktree on Vast; SHA-identical to pushed commit; overlay preflight.
3. Kill experiments 1–5 on CUDA (bounded smoke, disposable seeds — NOT the sealed split).
4. Seal fresh F5 split (generation-only, no checkpoint load).
5. Full arms A–D on the sealed split; aggregate per-fold receipts; bootstrap CIs (10k resamples, per-env paired deltas).
6. Verdict + governance seal; main untouched; `HENRI_F5_CODEC=0` until promotion approval.

---

## 9. Governance commitments

- This document SHA-256 sealed via `F5_PREREG_SEALED` audit event (child of `392a23ff`).
- Any pre-seal amendment changes the SHA: recompute, record both, re-seal — never edit silently.
- Promotion to main requires: all gates pass + explicit user approval (load-bearing change).
- Negative result seals as `FALSIFIED_AT_SCALE` (or the specific gate class) and keeps the carrier default-OFF — negatives are governance wins.

---

## 10. Next actions

1. [THIS DOC] seal + commit + governance event.
2. USER APPROVAL GATE: approve implementation (TDD harness + remote CUDA metric tests + fresh split + bank re-evaluation).
3. Implement `fpb_qfhrr_codec.py` + contract tests (RED→GREEN).
4. Remote kill experiments, fresh split seal, full arms, verdict.
