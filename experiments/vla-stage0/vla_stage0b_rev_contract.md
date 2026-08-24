# System-1 Stage-0b-rev — Frozen Nonlinear Encoder + Stage-0c-rev Audit — PRE-REGISTRATION

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding** · Status: SEALED BEFORE IMPLEMENTATION

## Upload provenance (proposal artifact — audited, not authoritative spec)
- `G:/My Drive/HENRI_Inbox/Stage-0c_Identifiability_Audit_and_Stage-0b-rev_Architecture.md`
- 1,061 B, SHA-256 `dcce2be3a25e79fa8aefc9072b81eb11fe0dce130dd4964387f4ca1b3b6f717b`
- Content: 14-line diagram. Mechanism named: `x(4D) → Random Fourier / Phase Modulation → Nonlinear RFF → Circular Conv Slot_i → Flat 6144D (PR ≥ 16) → EDMD UNBLOCKED`.
- NOT specified in upload (authored here, sealed): feature formula details, B/S shapes and distributions, seed, dtype, normalization, C1–C3 text, gate mechanics.

## Feasibility derivation (done BEFORE code; Reference 3 §2)
- Quadratic polynomial lift of 4 vars: uncentered dim 1+4+C(5,2)=15; centered usable rank ≤ 14 → PR ≥ 16 **impossible** → would be `FALSIFIED_BY_DIMENSION_BOUND`.
- The upload specifies Random Fourier / trigonometric features, NOT a polynomial lift. For `φ(x)=[cos(Wx̂+b); sin(Wx̂+b)]`, `W∈R^{m×4}`, the feature map is nonlinear; the linear span of N distinct samples is generically `min(N, 2m)`.
- With m=192 (384 features), N=171 calib → span up to 171 > 16 → **dimensionally feasible; the proposal is NOT rejected by the dimension bound.**
- Slot binding via circular convolution `z_i = φ ⊛ k_i` is LINEAR in φ; it preserves (inherits) the feature rank. Per-slot unit norm does not change the span.
- The genuine empirical risks are (a) trajectory correlation compressing the spectrum (PR falling below 16) and (b) κ16 exceeding 100 on correlated rows. Both are measured, not assumed.

## Frozen parameter specification (B, S)
- m = 192 frequencies; features dim 384 = [cos; sin].
- **Input standardization (frozen):** per-dim mean/std computed ONCE from the 171 calibration observations (first 10 episodes in LEXICOGRAPHIC FILENAME ORDER — seeds {101,1010,1111,1212,1313,1414,1515,202,303,404}; matches the sealed Stage-0c audit `ce697efd…` partition). Eval episodes (seeds {505,606,707,808,909}, 133 records) use the same frozen transform. No centering/scaling of features after RFF.
- **B** = RFF phase bias `b ∈ R^192`, `b_i ~ U[0, 2π)`.
- **W** = `W ∈ R^{192×4}`, `W_ij ~ N(0, 1)`.
- **S** = slot key matrix `k ∈ R^{16×384}`, `k_ij ~ N(0, 1)` (frozen VSA binding keys).
- **Seed:** `rng = np.random.default_rng(20260824)` (single generation run).
- Generation executed ONCE by `vla_stage0b_rev_params.py` → saves `vla_stage0b_rev_params.npz` (W, b, k, calib_mean, calib_std). The encoder LOADS the npz only; no runtime RNG.
- **Freeze hash:** full SHA-256 of the npz bytes recorded in the seal; the encoder asserts the npz SHA on load; cross-process equality = identical npz bytes.

## Split correction (2026-08-24, OBSERVED manifest)
- Earlier reports stated calib 171 / eval 133 (audit-log artifact from the prior audit's own pair construction).
- Measured manifest `54b7350a…` truth: calib (first 10 episodes) = **213** records, eval (last 5) = **91** records, total 304.
- All calibration-dependent quantities (calib_mean/std, per-action matrices, gates) use the 213-record calibration partition. The 91-record eval partition is untouched until after rank freeze (G4 evidence only).
- Contract SHA-256 changes from the pre-correction value; both values are recorded in governance (pre `823a98f0…`, post-seal below).

## Encoder contracts (Stage-0b-rev carrier)
- **C1 (default OFF):** without `HENRI_STAGE0B_REV_ENABLE=1`, `encode()` returns the input **byte-identical** (np.array_equal, no copy mutation).
- **C2 (zero trainable state):** numpy-only implementation; no `torch` import, no Parameter, no backward, no optimizer. No module state mutation after construction.
- **C3 (frozen cross-process hash):** the encoder output hash for a fixed input set is byte-identical across separate processes (guaranteed by loading the identical npz).
- **C4 (shape/geometry):** output `(1,16,384)` float32; per-slot unit-norm sphere `||z_i||₂ = 1` (error ≤ 1e-6). Canonical geometry = sphere; NO torus claim.
- **C5 (deterministic restart):** fresh process + same npz + same input → byte-identical output.
- **C6 (sensitivity):** over DEDUPLICATED real observations (chain-continuity duplicates removed), ≥ 99% of distinct pairs have L2 > 1e-3; zero identical-input collisions in the audit set.
- **C7 (non-collapse):** flat calib matrix SVD rank ≥ 16; min slot std ≥ 1e-3 (measured on calib).
- **C8 (no env/learner access):** encoder imports no wrapper, no gymnasium, no learner; no side effects.
- **C9 (provenance):** verification runs on the VERIFIED Stage-0a corpus (304 records, 15 episodes, manifest `54b7350a…`); encoder verification uses REAL observations, never synthetic arrays.

## Stage-0c-rev audit procedure (calibration only; held-out untouched for rank selection)
- Corpus: `vla_stage0c_corpus/` manifest `54b7350a…`; episodes sorted by seed ascending; **calibration = first 10 episodes (171 pairs); evaluation = last 5 episodes (133 pairs) — NEVER relabeled as calibration.**
- Per-action flat matrices (6144-D): X0 (obs_t, a=0), Y0 (obs_next, a=0), X1 (obs_t, a=1), Y1 (obs_next, a=1).
- Metrics per matrix: participation ratio PR=(Σs²)²/Σs⁴, numerical ranks r(>1e-3·s1), r(>1e-6·s1), condition κ16 = s1/s16 (raw SVD, no regularization, no centering beyond the frozen input transform).

## Gates (combined, pre-registered)
- **G1:** PR ≥ 16 on EACH of X0, Y0, X1, Y1 (per-action floor, not pooled).
- **G2:** κ16 ≤ 100 on X0 and X1 (design matrices, raw singular values).
- **G3:** sample support: N_a ≥ 4r for BOTH actions, r ∈ {4, 8, 16, 32}. Freeze the LARGEST r passing G1–G3; if none → `IDENTIFIABILITY_BLOCKED`.
- **G4 (evidence only, NOT a gate):** after rank freeze, held-out reconstruction relative error on the 133 untouched pairs, reported but not used to select r.
- **G5:** NO adapter construction in this carrier. Stage-0c adapter requires a separate sealed pre-registration + explicit authorization.

## Verdict chain
1. Feasibility bound < gate → `FALSIFIED_BY_DIMENSION_BOUND` (no implementation, no audit run). → NOT triggered (feasible).
2. Encoder contracts fail → `ENCODER_REV_CONTRACT_FAILED` (no Stage-0c-rev audit).
3. G1–G3 fail → `IDENTIFIABILITY_BLOCKED` (audit artifacts preserved; no Koopman adapter).
4. All gates pass → `RANK_SELECTED` with frozen r; Stage-0c adapter remains separately gated.
- **VLA gate stays 0/12 regardless.** Passing this carrier proves bounded CartPole dynamics identifiability only — not perception, policy, continuous learning, or benchmark progress.

## Kill criteria (pre-registered)
- PR < 16 on any per-action matrix → BLOCKED (no re-tuning of W/σ to chase the gate; the frozen parameter spec is the experiment).
- κ16 > 100 → BLOCKED (conditioning failure on the raw design).
- Any encoder contract failure → STOP before audit.
