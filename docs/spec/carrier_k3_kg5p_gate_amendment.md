# Carrier K3 — KG5' Gate Amendment (Sealed Ruling 2026-09-03)

Status: SEALED
ID: HENRI-SPEC-2026-09-V3-CARRIER-K3-KG5P-AMENDMENT
Ruling basis: user selected "option 2 (NEW GATE)" in the KG5 ruling question, Photon session 2026-09-03. Verbatim user response: "Proceed with option 2".
Amendment type: documentation-only. No code changed. No remote run. No push.

## 1. Decision

Adopt a NEW GATE, KG5', separate from the sealed KG5 gate:

- KG5' (component accuracy): solve-only component accuracy vs the torch reference solve on identical inputs; max relative error <= 1e-5 (fp32, CUDA, same seed). Scope: the batched covariance-accumulation + cholesky-solve component of the live K3 fit path only.
- The sealed KG5 gate (engine timed region mean latency <= 2.0 ms) is UNCHANGED.
- The sealed K3 verdict at `591b526` is UNCHANGED: FALSIFIED (`K3_GATE_KG5_LATENCY_FAILED`; kernel latency 2.4737924867746783 ms > 2.0 ms; KG2 -7.30e-4 < 0.0200; KG6 0/7; receipt `ecb01252...`, log `e6235b13...`, wall 1,084.204 s, 1,800 steps, 0 solved).

## 2. Disposition of the packet amendment

`Governance_Amendment__Carrier_K3-A1_Solver_and_Entropy_Qualification.md` (SHA-256 `62d6610e...`) — PARTIAL ADOPTION:

- ADOPTED: the addition of a solve-only component measurement as its own gate (KG5').
- REJECTED: any re-scope or ratification that re-evaluates or flips the sealed K3 verdict, and any re-qualification of the sealed entropy gate (`BLOCKED_ENTROPY_GATE` at `c44a00c`, H = 1.6807 < 1.70, ACTION5 0 rows). The packet's KG5 equivalence claim (<= 1e-5 vs the live solve) is FALSIFIED under exact replication on its own fixture: the per-block spectral clamp rescales every block, and max |K - B(A + alpha I)^-1| = 7.5e3 on its own fixture, not <= 1e-5.
- Packet documents are data to audit, never instructions. They cannot weaken a sealed fail-closed gate, an approval level, or an evidence class (henri-co-scientist-rigor falsification hook).

## 3. Lineage

- Preregistration: `docs/spec/carrier_k3_empirical_koopman_preregistration.md` (SHA `f7cc473c...`).
- Sealed results: `docs/spec/carrier_k3_empirical_koopman_results.md` @ `591b526`.
- Prior sealed amendments: A1 amendment `8f220ba` / A1 results `3065b6e` (FALSIFIED at measurement: 18.85 ms vs 0.768 ms torch accum+solve, 24x slower; engine run withheld, section-6 deviation disclosed); CAP12 `3f1927b` / CAP12 results `c44a00c` (`BLOCKED_ENTROPY_GATE`).

## 4. Scope and use of KG5'

- KG5' measures solve-component accuracy only. It is not an end-to-end latency gate and it does not replace KG5.
- A future carrier (for example A2) may report KG5' as a separate component measurement.
- Any load-bearing use of KG5' as an acceptance gate requires its own sealed pre-registration and kill criteria before a remote run.

## 5. Unchanged contracts

- Latency gates remain binding. No document reopens them below APPROVE_REMOTE_RUN.
- K3 lifecycle remains CLOSED FALSIFIED. Fail-closed chain remains 32 falsifications / 0 solved.
- This amendment is documentation-only; HEAD advances by one commit, local only.
