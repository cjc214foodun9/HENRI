# Carrier K3 — KG5' Measurement Instrument (Sealed Pre-Registration)

Status: SEALED (pre-registration)
ID: `HENRI-SPEC-2026-09-V3-CARRIER-K3-KG5P-INSTRUMENT`
Branch: `feat/carrier-k3-empirical-koopman` (HEAD `80877b1` pre-seal)
Seal date: 2026-09-03
Approval basis: user message "Proceed with option b please build the kg5 instrument" (Photon session 2026-09-03).
Amendment source: `docs/spec/carrier_k3_kg5p_gate_amendment.md` @ `51273a0` (SEALED).

## 1. Purpose and scope

KG5' measures solve-component accuracy ONLY, per the sealed amendment `51273a0`:

- Tolerance: max relative error <= 1e-5 (fp32, CUDA, same seed).
- Reference: the torch reference solve on identical inputs.
- Scope: the batched covariance-accumulation + cholesky-solve component of the
  live K3 fit path only (pre-projection).
- The sealed KG5 gate (engine timed-region mean latency <= 2.0 ms) is UNCHANGED.
- The sealed K3 verdict @ `591b526` (FALSIFIED) is UNCHANGED.
- This instrument authorizes BUILD and local verification only. Remote
  execution requires a separate sealed human decision (APPROVE_REMOTE_RUN or
  higher).

## 2. Measured path (OBSERVED, live code)

`HENRI V2/experiments/verification/arc_k3_koopman_generator.py`
`BlockRidgeKoopmanFit.fit` (lines 192-252):

1. `A = einsum("nmi,nmj->mij", Xf, Xf)`; `B = einsum("nmi,nmj->mij", Yf, Xf)`.
2. `Aa = A + alpha*I`; alpha starts at `K3_ALPHA = 1e-4`; on cholesky failure
   alpha doubles (max `K3_ALPHA_MAX_DOUBLINGS = 4`); counted pinv fallback.
3. `L = cholesky(Aa)`; `K = cholesky_solve(B^T, L)^T` (the measured solve).
4. Contractive projection AFTER the solve (excluded from KG5' by scope; the
   per-block spectral clamp is the documented 7.5e3 falsification, sealed
   results doc line 21 @ `591b526`).

Probe imports the LIVE class; no reimplementation of the measured component.

## 3. Fixture corpus (pre-registered)

True operator per fixture: `K_true = scale * Q` with Q from QR of CPU-seeded
`randn(M, 8, 8)` (orthogonal), transferred to CUDA.
Rows: `X [n, 8192, 8]` unit-norm per block, fp16 ring storage then fp32 read
(mirrors `K3RingAccumulator.ordered()`); `Y = einsum K_true X`.

| Fixture | Scale | Rows | Seed | Skew | Role |
|---|---|---|---|---|---|
| F1 | 0.5 | 64 | 20260906 | none | verdict (no-fire) |
| F2 | 0.5 | 128 | 20260907 | last col x0.25 | verdict (no-fire, skew) |
| F3 | 0.9 | 248 | 20260908 | none | verdict (no-fire, near bound) |
| F4 | 1.4 | 64 | 20260909 | none | boundary control (MUST fire) |

Seed base 20260905 (per-fixture seeds +1..+4 as listed).

## 4. Reference solve

`torch.linalg.solve(Aa, B^T).T` on bitwise-identical `Aa`, `B` einsum systems
(same device, dtype, order). Alpha in the reference = the alpha actually used
by the fit after doublings. Independent numerical path (LU) vs the live
cholesky_solve path.

## 5. Error metric and verdict

- Per-block Frobenius relative error: `||K - Kref||_F / ||Kref||_F`.
- Verdict statistic: max over blocks and over verdict fixtures F1-F3.
- PASS if verdict statistic <= 1e-5.
- Fixture-validity controls (fail-closed, checked BEFORE the tolerance):
  F1-F3 `fired_blocks == 0` (no-fire regime => returned K equals the raw
  solve output); F4 `fired_blocks > 0` (projection engaged; proves fixture
  sensitivity); no `pinv_fallback` in F1-F3; max kappa(Aa) <= 1e2 across
  F1-F3 (fp32 factorization floor).
- Verdict symbols: `KG5P_GATE_ACCURACY_PASS` | `KG5P_GATE_ACCURACY_FAILED` |
  `KG5P_BLOCKED_INSTRUMENT` (any control failure; precedence fail-closed).

## 6. Probe identity (pre-seal pinned)

- Probe: `k3_kg5p_solve_acc.py`, SHA-256
  `46eda498e4a3cc8e71d46d2b16dbb791a3e18a377a8cb0c00135b81809602e44`,
  5,592 B, 156 lines, LF-only. Remote path `/tmp/kg5p/` on the dispatch tree.
- Smoke: `kg5p_smoke.py` (CPU, M=64; import-real-probe; asserts gate math).

## 7. Ship gate and run gate

- Ship gate: local CPU smoke PASS (`KG5P_SMOKE_OK`). OBSERVED 2026-09-03 at
  M=64: F1 1.68e-7, F2 1.92e-7, F3 1.61e-7 (all <= 1e-5, no fire); F4 fired
  64/64 (control engaged); kappa 2.0-25.5 <= 1e2. Smoke values are plumbing
  evidence only; gate values come from the CUDA M=8192 run.
- Run gate: remote CUDA run on the idle RTX 5090 requires APPROVE_REMOTE_RUN
  or higher. Not authorized by this instrument.
- Exclusions: engine timed run, KG5 latency re-measurement, KG1-KG4 re-runs,
  and any verdict flip on sealed KG5/K3 are OUT OF SCOPE.

## 8. Governance

Doc-only atomic commit on this branch, local only, no push. Untracked
`henri_audit_chain.json` preserved. Negative outcome (accuracy FAILED or
instrument BLOCKED) is a governance result with this artifact as the record.
