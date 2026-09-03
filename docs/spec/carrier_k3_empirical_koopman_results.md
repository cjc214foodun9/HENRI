# Carrier K3 — Empirical Block-Koopman Generator: Sealed Results

Document Identifier: `HENRI-SPEC-2026-09-V3-CARRIER-K3-SEALED-RESULTS`
Date: 2026-09-02
Branch: `feat/carrier-k3-empirical-koopman`
Build commits: `14ae2b4` (K3 build + sealed prereg), `ddc950f` (goal-adapter SHA-test reconciliation)
Base: `2f9bc57` (Carrier C1 closeout)

## Artifact SHAs

| Artifact | SHA-256 |
|---|---|
| Supplied prereg (inbox) | `841ac581…` (15,459 B, 203 lines) |
| Supplied fused Triton kernel (inbox) | `bff01749…` (6,205 B, 155 lines) |
| Sealed prereg (repo, `docs/spec/carrier_k3_empirical_koopman_preregistration.md`) | `f7cc473c42c68e474928711dd127014090bd5dd5a1cdf85678b36691d97a4004` |
| Dispatch instrument (inbox) | `98ef866a…` (`HENRI-AUTH-2026-09-V3-CARRIER-K3-DISPATCH`, AUTHORIZED FOR DISPATCH) |
| Run receipt (remote → local) | `ecb01252b56b701d2312955c6e29c5d7531a67e2a5a131de9bd1235481791270` |
| Run log (remote → local) | `e6235b135177579676696f82bce09d4167e1c9eff5ded186696130478967da32` |

Machine: vast-5090 (RTX 5090, instance 47411800), detached worktree `/workspace/henri-k3-dispatch` @ `ddc950f`, clean.
Cohort: 12 envs, seed `20260930`, 1,800 steps (150/env), 1,084 s wall, EXIT 0.
Policy mode: `K3_EMPIRICAL_KOOPMAN` (`HENRI_K3_KOOPMAN=1`).

## Verdict

**K3_FALSIFIED — 31st sealed falsification in the chain.** Verdict symbol fired by
fail-closed precedence: `K3_GATE_KG5_LATENCY_FAILED`; the load-bearing scientific
falsifications are KG2 (action coupling ≪ bound on the 7-env seal basis) and KG6
(0/7 solved). W0 stays gated. A negative result is a governance win; the record is
the artifact.

## Gate table (live receipt values)

| Gate | Bound | Live value | Result |
|---|---|---|---|
| KG1 | held-out one-step rel. err ≤ 0.1500 | 0.0785 (12 samples) | PASS |
| KG2 | `mean_delta_nu_wp` over goal-available envs ≥ 0.0200 | **−0.0007300** | **FAIL (FALSIFIED)** |
| KG3 | min pairwise operator separation ≥ 0.0500 | None (0 samples) | NOT_ENGAGED (no pairwise samples emitted) |
| KG4 | post-scale ρ enforced ≤ 1.000001; engagement reported | σ_raw_max 1.7630, σ_post_max 1.0000031, fired blocks 88,353 | ENFORCED; engagement OBSERVED |
| KG5 | local score-path + refit CUDA-event mean ≤ 2.00 ms | **2.4738 ms** | **FAIL** |
| KG6 | envs solved (goal-available subset) ≥ 1 | 0 | **FAIL** |
| — | W0 (`WavePacketPathSearch`) | — | stays GATED (KG2 unmet) |

`mean_latency_ms` 302.6 is the remote arcade round trip, reported separately and
NOT the KG5 basis (prereg §5 amendment; C1 LG3 spurious-flag precedent).

## Per-env Δν (observed)

Seal basis (7 goal-available): `ar25 +1.37e-4, bp35 −7.07e-3, cd82 −5.18e-5,
ft09 0.0, g50t +1.52e-3, ka59 0.0, lp85 0.0` → mean **−7.30e-4**.
Bank-void diagnostic (5 envs, excluded from seal basis by prereg §5.6):
`cn04 +3.98e-5, dc22 −4.12e-4, lf52 −3.75e-5, ls20 −3.82e-4, m0r0 +5.96e-4`.
All envs: 0 levels, 0 waypoint advances, 17 resets, 336 creeps, 1783 ring pushes,
137 fit calls, 980 K3 score calls, 2.47 ms local mechanism mean.

## Disclosed limitations

1. KG5 measured 2.47 ms on the batched-torch solve path — the honest measured
   number replacing the prose 45 µs projection (audit deficit #1 CONFIRMED live).
   In-SRAM Triton Cholesky is the bounded next carrier; it amends the sealed
   prereg → REQUIRES_APPROVAL.
2. KG2/KG6 seal basis is 7 envs, labeled CONDITIONAL vs C1's 12-env basis; the 5
   bank-void envs need a fresh capture before a full 12-env seal basis (audit
   deficit #2; REQUIRES_APPROVAL).
3. No parameters tuned post-seal. C1's sealed FALSIFIED record untouched.
