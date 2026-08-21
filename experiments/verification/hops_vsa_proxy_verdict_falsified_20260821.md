# HOPS-VSA Proxy Verdict — FALSIFIED (Class 4.5)

**Doc ID:** HENRI-CLASS45-HOPS-PROXY-FALSIFIED-2026-08-21
**Date:** 2026-08-21
**Packet:** `experiments/verification/hops_vsa_proxy_gate_20260821.md` (HENRI-CLASS45-HOPS-PROXY-GATE-2026-08-21)
**Branch:** `hops-vsa-proxy` (base `5652c41`; Path B2 sealed `479dba4`)
**Evidence:** `experiments/verification/hops_vsa_proxy_evidence/proxy_smoke_d4096_cpu.jsonl` SHA `a293b9a0a3000854ef489dfc8bf627d8ceddee936722c7077f1b0daed619239e`

## Verdict

**HOPS_PROXY_FALSIFIED.** The kill fired at the cheapest stage (local smoke,
D=4096, CPU) because every failed mechanism is dimension-invariant: the
failures are structural, and D=65,536 concentration only shrinks null-channel
cosines further. No CUDA suite, no external gate (pre-registered).

## Measured gates (OBSERVED, live probe run 2026-08-21)

| Gate | Criterion | Measured | Verdict |
|---|---|---|---|
| P1 carrier engagement | removed-energy ≥ 8× random baseline | 0.001854 vs random k/D 0.001953 → **0.95× random**; residual 0.9991 | FAIL (no structured carrier; at D=65,536 absolute bar 1e-3 vs baseline 1.22e-4 also fails) |
| P2 paired rank/margin | rank ≤ 5 AND margin ≥ 0.25 both targets, no regression | /23: control r39/m−0.036 → treatment r3*/m−0.038; /35: r16/m−0.019 → r36/m−0.018 | FAIL (margins negative, /35 regressed) |
| P3 veto discrimination | veto fraction ∈ [0.05, 0.95], oracle not vetoed | fraction **1.0**, oracle vetoed BOTH targets | FAIL (all-veto = inert gate) |
| P4 invariants | gram_error ≤ 1e-6, finite, thin V, pool=71, one oracle | gram_error 1.79e-6; pool 71/71; oracle 1/1; V [4096,8] | FAIL (gram) |

* r3 is a degenerate artifact: with veto fraction 1.0 every treatment score is
  −1e9, so ranking collapses to original pool index (oracle sat at index 3).

## Root cause (DERIVED / structural)

1. Decoder unit-normalized random-ring waves are quasi-orthogonal at high D;
   an 8-dim label-derived skeleton removes only ~k/D energy — measured 0.95×
   the chance baseline. The basis does NOT span the production qFHRR carrier
   (the open caution in `hops-vsa-reference-core-lessons.md` is now resolved:
   FALSIFIED).
2. Null-channel residual cosines are ~0.00x, so the Sagnac veto
   (Δ = 1 − cos > 0.35) fires on every candidate — the all-veto degeneracy.
3. A +0.25 margin in the null channel is unattainable: all null cosines sit
   in a ~0.03-wide band.

## Dispositions

- `--hops-vsa-rank` stays **default-OFF**; module not promoted, not deleted.
- Runner latent-crash fix (remove `ring_to_real_wave` on `decoder._wave`
  float32 output; `.to(device)` only) is committed with this seal — it
  corrects a committed default-OFF path that would crash on CUDA.
- No main push; `5652c41` remains the release candidate at the approval
  boundary. No promotion of any HOPS component.
- Path B2 remains CLOSED (Gate A FALSIFIED, Gate B skipped, seal `479dba4`).
  Reopens only with a new pre-registered packet + user approval.
- Supplied `Universal_Zone_C_TimescaleDB_Seeding_Module.py`: audited only,
  never executed; credentials REQUIRES_APPROVAL (unchanged).

## Claim records (CoE)

| claim_id | status | evidence |
|---|---|---|
| HOPS-P1 carrier engagement fails | FALSIFIED | smoke receipt (removed 0.001854, baseline 0.001953) |
| HOPS-P2 rank/margin fails | FALSIFIED | smoke receipt (/23 m−0.038, /35 m−0.018, r36) |
| HOPS-P3 veto inert (all-veto) | FALSIFIED | smoke receipt (fraction 1.0, oracle vetoed) |
| HOPS-P4 gram fails | FALSIFIED | smoke receipt (1.79e-6 > 1e-6) |
| HOPS-RUNNER fix (latent CUDA crash) | OBSERVED | py_compile + smoke; branch diff vs `5652c41` |
| D-invariance of failure | INFERRED | quasi-orthogonality + concentration argument; cheapest-kill executed |

## Next actions (bounded)

1. HOPS-family reopens only with a NEW pre-registered packet: basis must be
   learned from production waves (not label-derived), veto must operate on a
   channel with measurable cosine spread, margin mechanism must be specified.
2. Else continue the audit-map remediation list (API-bridge telemetry,
   decoder memorization, egress boundary, `compile_functor` re-gate).
