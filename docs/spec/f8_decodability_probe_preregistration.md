# Carrier F8 Pre-Registration — Direct Supervised Linear-Decodability Probe

**Document ID:** HENRI-SPEC-F8-DECODABILITY-PROBE
**Directive:** HENRI-DIR-2026-08-F7-POSTMORTEM-DECODABILITY-PROBE (SHA-256 `4a5a386b3d13051a8937645033b08e1b41a863de64e4ad6c1881693a57ade394`, 18,387 B)
**Ratified verdict:** `F7_GATES_VERDICT #177c30b5… = FALSIFIED_AT_SCALE` (ledger record 1,028; chain head `e2622541…`, 1,030 records)
**Branch target:** `carrier/f8-decodability-probe`
**Ledger target:** record 1,031+
**Host substrate:** Vast RTX 5090 (CUDA 13.0 / PyTorch 2.12)
**Bank:** F3 v2 trajectory bank, 1,536 rows, 12 envs, 7 actions; npz `9e3c01b4…`, jsonl `1ca089b2…` at `/root/f3-run/telemetry/f3_bank_capture_v2/`

---

## 1. Academic foundation

F4–F7 evaluated four operator families (random-ring 0.4414, FPB codec 0.4352, unitary circulant 0.4360, non-unitary affine 0.4051) against the same bank. All plateau at the action-marginal. F7 proved the affine cannot reconstruct its own 20 demo rows (G1 = 0.25 ≪ 0.99) despite D ≫ M over-parameterization. Conclusion chain:

- If raw bank waves `Ψ ∈ ℂ^{1536×65536}` carried a linear action channel, min-norm least squares would interpolate the 1,536 labels (D ≫ N regime).
- F7's failure at M=20 per env implies the within-env demo waves are near-collinear across actions.
- Hypothesis H0: `I(Ψ_t; a_t) ≈ 0` — the bank encodes static visual geometry, not motor intent.
- Hypothesis H1: some unconstrained supervised model decodes actions above the majority baseline.
- Carrier F8 is a diagnostic on the representation, not a production mechanism. It trains no production path, changes no runner, and is default-OFF.

## 2. Probes (directive §3 verbatim family)

| Probe | Model | Features | Notes |
|---|---|---|---|
| P1 | Unregularized linear least squares | real/imag concat of Ψ (2D real) | D ≫ N ⇒ unique solution is the min-norm (Moore–Penrose) fit; implement via thin SVD + tiny ridge λ=1e-6 for numerical stability (disclosed correction 2) |
| P2 | Multinomial logistic regression | real/imag concat (2D real) | Cross-entropy + L2 sweep λ ∈ {1e-6, 1e-4, 1e-2, 1e-1}; Adam lr 1e-3, ≤200 epochs, early stop on val fold |
| P3 | 3-layer MLP: 131,072 → 1024 → 256 → 7, GELU, LayerNorm | real/imag concat | Adam lr 1e-3, ≤200 epochs, early stop; dropout 0.1 between hidden layers |
| P4 | k-NN (k=1, 3) | complex Hermitian distance `d(Ψ_i,Ψ_j) = 1 − |(1/D)⟨Ψ_i,Ψ_j⟩|` | exact directive formula; k=1 train accuracy is 1.0 by self-hit — excluded from G1 (disclosed correction 3) |
| TD | Temporal difference: `ΔΨ_t = Ψ_{t+1} ⊛ Ψ_t†` (elementwise conj-product), same 4 probe families applied | pairs within env segments only, never across env boundaries | tests G4 |

## 3. Cross-validation protocol

- 10-fold stratified CV across all 1,536 rows. Stratification key = action label (directive: "stratified on ALL 1,536 rows").
- Folds are ROW-disjoint; index lists recorded per fold in the receipt.
- **TD paired-row constraint (disclosed correction 4):** a TD test row t is valid only when row t+1 also lies in the same test fold (otherwise the successor leaks train information). G4 compares `Acc(ΔΨ)` vs `Acc(Ψ_static)` on the IDENTICAL valid row subset per fold. Report both n and accuracy.
- **Trivial-env diagnostic (disclosed correction 5):** envs `lp85`/`ft09` are single-action loops (H(A) ≈ 0) and any majority predictor scores 1.0 there. Global gates run verbatim per directive, AND the receipt reports non-trivial-env CV accuracy separately (excluding lp85/ft09) so the global number is never misread as skill.
- No new heldout split is created or consumed: the directive mandates CV on the full pinned bank. Bank hashes are re-verified before the run.

## 4. Pre-registered gates (directive §4 verbatim)

| Gate | Requirement | Kill |
|---|---|---|
| G1 | Full-bank in-sample train accuracy ≥ 0.9500 (parametric probes P1/P2/P3) | K1: intrinsic linear inseparability |
| G2 | 10-fold CV Top-1 accuracy ≥ 0.6000 | K2: generalization absence |
| G3 | `Acc_CV − P(a_majority) ≥ +0.2500` | K3: information deficit in static phase vectors |
| G4 | `Acc(ΔΨ) − Acc(Ψ_static) ≥ +0.2000` (paired rows) | diagnostic: action lives in derivative |

Majority baseline `P(a_mode)` computed from the bank label marginal at runtime and recorded in the receipt.

## 5. Verdict taxonomy (ternary, pre-registered)

`Acc_max = max over probes of CV accuracy` (TD arm reported separately):

- **CASE A — `F8_PROVEN_NO_ACTION_SIGNAL`:** `Acc_max ≤ P(a_majority) + 0.05`. Supports H0. Directive's stated root cause: ingress encodes static state, not `ΔΨ` or goal direction. Sanctioned next step per directive: re-engineer ingress to transition/derivative encoding (separate directive required).
- **CASE B — `F8_DECODABLE_SIGNAL_EXISTS`:** `Acc_max ≥ 0.75`. Supports H1; prior in-situ compilation was sample-starved; sanctioned pivot: supervised egress (separate directive required).
- **CASE C — `F8_INDETERMINATE`:** `0.30 < Acc_max < 0.75` (or gates mixed). Partial signal exists; not attributable to a single root cause; next falsification proposed in the verdict report.

Gates G1–G4 do not independently determine the verdict; they attribute the failure stage. The verdict is determined by Acc_max relative to the CASE thresholds.

## 6. Disclosed corrections (all pre-seal)

1. **Complex features:** bank waves are complex. Linear/logistic/MLP consume `concat(Re Ψ, Im Ψ)` ∈ ℝ^{131072} (2D real). k-NN uses the directive's complex Hermitian distance directly.
2. **"Unregularized" LS:** D ≫ N (65,536 ≫ 1,536) ⇒ unregularized LS is non-unique; the reproducible unique solution is min-norm LS. Implemented as thin-SVD pseudo-inverse with λ=1e-6 ridge for stability; this is the standard over-parameterized interpolant the directive's §1.2 argument presumes.
3. **G1 scope:** k-NN k=1 train accuracy is 1.0 by self-match (vacuous); G1 is measured on P1/P2/P3 only. k-NN contributes to Acc_max via CV accuracy.
4. **TD paired rows:** see §3. Unpaired TD test rows are excluded and counted.
5. **Trivial envs:** see §3.
6. **Memory feasibility:** complex bank 804 MB fp32; real/imag feature matrix 1.6 GB; P3 first layer 134 M params (537 MB); per-fold SVD cost O(N²·D) ≈ 3e11 flops ≈ minutes on 5090. All bounds within a single 32 GB GPU with no [D,D] materialization (max dense [N,D] = 1.6 GB).

## 7. Execution order (directive §5)

1. Seal this spec (`F8_PREREG_SEALED`) with directive hash + spec hash + bank hashes.
2. RED: contract tests `tests/contract/test_f8_decodability_probe.py` (toy D=64, planted linear structure; see §8) — fail against absent module.
3. Implement `HENRI V2/arc_f8_decodability_probe.py` (P1–P4 + TD arm; `HENRI_F8_PROBE=1` gate; bank loader with schema validation; 10-fold stratified CV; receipt writer).
4. GREEN: F8 contract suite + full regression (F7+F6+F5+arc).
5. Seal `F8_IMPL_COMMITTED` with file hashes; commit + push branch.
6. Remote CUDA gauntlet on Vast 5090 at exact SHA: bank hash re-verify, probes P1–P4 + TD, gates G1–G4, receipt.
7. Seal `F8_GATES_VERDICT` with the ternary CASE label; ledger verify; final report.

## 8. Contract tests (toy-scale, CPU)

Synthetic fixtures with PLANTED structure (never the production bank):

- C1: P1 recovers planted linear labels (train acc ≥ 0.99, toy D=64, N=256, 3 classes).
- C2: P2 separates a separable synthetic set (CV ≥ 0.95).
- C3: P4 (k-NN) beats majority on clustered synthetic data.
- C4: TD arm — plant `a_t = f(Ψ_{t+1} − Ψ_t)`; TD probe must beat static probe by the toy analog of G4 (≥ +0.20).
- C5: default-OFF differential — probe code raises/no-ops without `HENRI_F8_PROBE=1`.
- C6: bank loader validates schema (shape, dtype, label range 1..7, env segment ids) on a tiny synthetic npz.
- C7: fold index disjointness (train ∩ test = ∅, union = all rows).
- C8: per-fold class coverage recorded; every fold contains ≥ 2 distinct action classes (else the fold is flagged, not silently dropped).

## 9. Honest limits

- CV on the pinned bank measures decodability of THIS bank's representation. It does not validate any production mechanism.
- Acc_max ≥ 0.75 (CASE B) does not license wiring supervised egress into the runner; that requires a new directive.
- The trivial-env disclosure means global numbers above may overstate; the receipt's non-trivial-env column is the interpretable figure.
- Verdicts are `OBSERVED` only from the remote CUDA receipt; local CPU contract passes are code-health evidence only.
