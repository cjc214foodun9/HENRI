# System-1 Stage-0c — Identifiability Premise Audit (2026-08-24)

**Verdict: IDENTIFIABILITY_BLOCKED** · Reference 3 (gpt-5.6-sol) binding
No adapter constructed. Stage-0c NOT authorized.

## Inputs
- **Corpus:** `vla_stage0c_corpus/` — 15 episodes, distinct seeds {101…1515}, 304 real records, manifest SHA `54b7350a…`; built through the VERIFIED Stage-0a wrapper (provenance `eb1a061c…`), complete `(obs_t, action, obs_next)` records with episode boundaries.
- **Encoder:** Stage-0b frozen (init hash `3527b7fd…`), enabled via `HENRI_STAGE0B_ENABLE=1`.
- **Split:** calibration = first 10 episodes (171 pairs), held-out = last 5 episodes (133 pairs). Basis would be fit on calibration ONLY.
- **Lift design:** separate per-action operators K_0, K_1 (additive action embeddings REJECTED, corpus #23 — crosstalk).

## Spectra (calibration partition, flat 6144-D encodings)
| Matrix | n | PR | r(>1e-3·s1) | r(>1e-6·s1) | κ16 |
|---|---|---|---|---|---|
| Zt_state | 171 | 1.8 | 12 | 56 | 1.9e3 |
| Y_successor | 171 | 1.7 | 12 | 55 | 2.2e3 |
| X0_action0 | 102 | 1.8 | 12 | 50 | 1.9e3 |
| Y0_action0 | 102 | 1.5 | 10 | 44 | 3.3e3 |
| X1_action1 | 69 | 1.9 | 11 | 45 | 2.5e3 |
| Y1_action1 | 69 | 1.8 | 12 | 45 | 2.2e3 |

## Gates (pre-registered in the audit script)
- Candidate `r ∈ {4, 8, 16, 32}` requires `r < PR_floor` AND `N_a ≥ 4r` for BOTH actions.
- a0: floor 1.5, N=102 → no candidate. a1: floor 1.8, N=69 → no candidate.
- `basis_rank_chosen = 0`. Audit JSON SHA `ce697efd…`.

## Structural cause
The Stage-0b encoder computes `base = obs @ W1` (a linear map, rank ≤ obs_dim = 4), then a rank-1 slot modulation `raw = base * G + B`. The output space is therefore a 4-dimensional linear image modulated per slot — intrinsic rank bounded by 4. Trajectory-correlated CartPole states compress the reachable set to an effective participation ratio of ≈1.5–1.9. A rank-4-bounded observable cannot support a rank-r Koopman fit for r ≥ 4 with N ≥ 4r.

Note: r=1 would pass the numeric gates but is a scalar gain, not a CartPole dynamics operator; the pre-registered candidate set deliberately excluded it.

## Rank-80 correction (Stage-0b telemetry)
The Stage-0b "SVD rank 80" was computed on the **per-slot stack** (55 distinct observations × 16 slots = 880 rows × 384 cols), NOT the flat per-observation matrix (55 × 6144). Ordinary matrix rank cannot exceed the row count; both claims are now documented exactly. Flat effective rank here ≈ 12 @ 1e-3, PR ≈ 1.8.

## Decision
- **NO adapter construction. C1–C12 pre-registration NOT sealed** (no valid r to freeze).
- Next options (require user decision):
  - (A) Replace the Stage-0b encoder with a nonlinear higher-rank encoder on the same boundary (new Stage-0b carrier revision, own contracts + verification).
  - (B) Choose a richer dynamical substrate.
  - (C) Hold.
- **VLA gate remains 0/12.** Stage 1+ stays an architecture map only.
