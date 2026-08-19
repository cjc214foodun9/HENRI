# Phase 8.32 — Calibrated Action Head (Stiefel-Ridge) Pre-Registration

## 1. Source
- Directive 2026-08-19: `G:\My Drive\HENRI_Inbox\Calibrated_Action_Head_Module.py` (8,790 B)
  sha256: `db5c91df7d72fc5345edf658cf03f6aaef02620edf057b5e0a9806e87e4517ce`
- Docstring identity: `henri_calibrated_action_head.py`
- Mechanism: UWE_Ingress Ψ_t (D=65,536) → h_t = GELU(LayerNorm(W_down Ψ_t)) ∈ R^2048
  → W_act* = argmin ||A − W H||²_F + γ||W||²_F → Stiefel retraction W = U Vᵀ
  → gate E_cal ≤ 0.05 AND Δ_sagnac ≤ 0.20 → default-ON; else default-OFF.
- Honest corrections over inbox module (audit 2026-08-19):
  (a) HELD-OUT gate (inbox module fitted AND evaluated on the same samples and
      asserted ON — mock loop + leakage; this build gates on held-out only);
  (b) SVD-form ridge (solve(HᵀH+γI) unstable at M < L; SVD form stable);
  (c) `sagnac_stress` labeled proxy (action-space L2, NOT wave homodyne);
  (d) synthetic fixture can never activate production (`ACTION_HEAD_SYNTHETIC_ONLY`).

## 2. Authorized-data boundary (honest)
- The uploaded module's __main__ SYNTHESIZES trajectories and asserts ON → mock loop,
  never capability evidence. Gate fires ONLY on authorized held-out trajectories.
- ARC arcade exposes no authorized (o_t, a_t, o_t+1) tuples (run5: BLOCKED_NO_DEMOS ×19)
  → production transition = BLOCKED_NO_ACTION_TRAJECTORIES.
- Self-play telemetry (own run jsonl) is action-state correlation only (SANS boundary),
  NOT task semantics; eligibility still requires external task outcomes with head active.
- This commit: mechanism + gates + artifact, DEFAULT-OFF, zero production wiring.

## 3. Gates G1–G10 (pre-registered)
- G1: module imports clean; no production side effects; default-OFF flag HENRI_ARC_CALIBRATED_HEAD
- G2: no [D,D] allocation; W_down [2048, 65536] documented (536.9 MiB fp32); ridge via SVD form
  W = V·diag(s/(s²+γ))·UᵀA, stable for M < L; equivalence to torch.linalg.solve on M ≥ L fixture
- G3: Stiefel retraction orthonormality ‖W_retracted·W_retractedᵀ − I‖ ≤ 1e-5 (rows orthonormal, A < L)
- G4: held-out gate: E_cal ≤ 0.05 AND Δ_sagnac ≤ 0.20 on HELD-OUT only;
  in-sample-only fit must NOT qualify (leakage kill)
- G5: artifact `henri.calibrated-action-head.v1`: self-hash round-trip, weight sha256,
  dataset digest, split identity, action ordering, no-eval-cache provenance
- G6: score-eligibility dominance: artifact alone does NOT set score_eligible;
  requires production wiring + task validation; trained_action_head_active=false here
- G7: typed errors: M<2, action_dim mismatch, wave dim mismatch, unreadable artifact
- G8: local full suite (env-clean) passes
- G9: remote CUDA suite at exact tip passes, isolated, zero competing procs
- G10: synthetic self-qualification attempt = INVALID; any gate fail → default-OFF, no merge

## 4. Kill rules
- K1: synthetic fixture flips status to ON in any committed path → KILL (mock loop)
- K2: held-out gate passes on in-sample-only data → KILL (leakage)
- K3: ridge solve allocates [2048,2048] on CPU or OOMs at M<L fixture → KILL
- K4: artifact self-hash mismatch on load → KILL
- K5: remote CUDA bucket-4 checkpoint failure → preserve log, overlay verify, rerun SAME SHA

## 5. Deliverables
- `henri_calibrated_action_head.py` (repo root, flat import) + unit/contract tests
- prereg + ingest manifest in experiments/verification/
- Local env-clean suite → commit → push → remote CUDA suite at exact tip
- Honest report: mechanism OBSERVED on fixtures; production transition BLOCKED_NO_ACTION_TRAJECTORIES
