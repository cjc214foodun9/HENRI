# F2-M3 Calibrated Hopfield Lexical Egress Codebook — Pre-registration

**Spec ID:** SPEC-2026-08-29-F2-EGRESS
**Author:** /henri-architecture (Implementation Engine Holon), approved by user 2026-08-29
**Source authority:** Project HENRI VLA Convergence Gap Analysis (SHA-256 `81cd8a33401e03e1…`), Step 2 / Gap 4
**Carrier:** F2-M3 Hopfield Lexical Egress Codebook Calibration (default-OFF `HENRI_F2_EGRESS=1`)
**Host:** NVIDIA RTX 5090 (CUDA 13.0 / PyTorch 2.12 / Triton 3.7), remote `vast-5090`

---

## 1. Problem statement

The active codebase down-projects continuous wave states through an uncalibrated linear head
(`down_proj [2048,65536]` + `lm_head [32000,2048]`), producing near-uniform logits
(`BLOCKED_SEMANTIC_CAPACITY`, sealed K2/U2 2026-08-25). Gap 4 requires a **calibrated Hopfield
lexical codebook** M so that

```
z_clean = Softmax(beta * Re(Psi* M^dagger)) M ,  M in C^{|V| x D}
```

yields >99% syntactic precision on discrete action tokens (code primitives / ARC GameActions).

## 2. Mechanism (dual-ridge factorized codebook, dense-ban compliant)

- Codebook M = A·B with A = Yᵀ(X Xᵀ + λI)⁻¹ ∈ ℝ^{V×N}, B = X ∈ ℝ^{N×D}.
- X = provenance-pinned calibration wave stack [N, D] (trajectory bank npz), Y = one-hot token/action labels [N, V].
- Ridge λ = 1e-3 (tunable only pre-measurement; frozen at seal).
- Snap: `z_clean = softmax(beta * Re(Psi*M^dagger)) @ M`, beta = 8.0 (Hopfield snapping, Ramsauer 2008.02217).
- **No dense [D,D] tensor**: the solve uses thin-SVD `X = U S Vᵀ` → `A = Yᵀ U S⁻¹ Vᵀ` (dual form); peak memory ~ N·D float32.
- No-BPTT closed-form calibration (per the phase 8.32b trajectory-bank precedent); **zero trainable parameters** in the calibrated codebook.
- Accumulation dtype float32 throughout.

## 3. Data provenance (calibration)

- Source: trajectory bank `henri_trajectory_bank.py` (exists at `/workspace/gate1-run/HENRI V2/henri_trajectory_bank.py`).
- Calibration split ONLY; **no heldout exposure during calibration**.
- A3 gate: calibration-split only; evaluation consumes a separate held-out split with provenance hash.
- If trajectory bank lacks code tokens/action labels, calibration is `BLOCKED_MISSING_PREMISE` until an authorized manifest exists.

## 4. Gates (acceptance)

- **G1** (held-out P@1): ≥ 0.99 on the held-out split (single-pass retrieval precision).
- **G2** (syntactic validity): ≥ 0.99 on discrete action tokens (code primitives / ARC GameActions) — fraction of decoded tokens that parse / are legal actions.
- **G3** (margin over legacy): calibrated codebook beats the legacy linear head by ≥ +0.05 held-out P@1 on identical frozen waves.
- **G4** (default-OFF differential): with `HENRI_F2_EGRESS=0`, the full pipeline is byte-identical to pre-wire (differential, not flag-read).
- **G5** (provenance/eligibility boundary): codebook alone NEVER grants score eligibility; full checkpoint/provenance chain still required.

## 5. Kills

- **K1**: held-out P@1 < 0.99 → carrier KILLED, default-OFF preserved.
- **K2**: no ≥ +0.05 margin over legacy linear head on identical frozen waves → KILLED.
- **K3**: no engagement telemetry (no `f2_egress_status`, no consumed signal) → KILLED as mock loop.
- **K4**: any dense [65536,65536] allocation in the calibrated path → KILLED (dense-ban violation).

## 6. Telemetry (per-call, compact)

`f2_egress_status` (`ENGAGED`|`BLOCKED`), `f2_p1_heldout`, `f2_syntactic_validity`, `f2_margin_legacy`,
`f2_codebook_bytes`, `f2_beta`, `f2_ridge_lambda`, `f2_default_off` (bool differential).

## 7. Verification (remote CUDA only)

- Standalone suite: `tests/contract/test_f2_egress_codebook.py` (RED first, tolerances frozen in header).
- Synthetic + gradient-check commands in HarnessContract B; benchmark command (performance claim) included.
- Local CPU runs are software sanity only; **remote CUDA is the verification boundary**.
