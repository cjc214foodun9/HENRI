# System-1 v0.6.0-dev — Zone C Retrieval Read-Path: RETRIEVAL_NO_EFFECT (mechanism-order-invariant)

**Date:** 2026-08-24 (CUDA 5090, vast-5090, PID 398551)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen)
**Split:** dev6_v060 (DISPOSABLE, seed 61623, n=60, 13 families, disjoint 4+4)
**Arms (matched budget 64 / beam 64):** A token beam · B13 uniform CEGIS-first · C engram-sim re-ranked
**Supplied artifact audited:** HENRI_Zone_C_System-1_Live_Wiring___Audit_Suite.py (sha `7d574e50`, 17,614 B; daemon skipped `.py` — SUPPORTED=.pdf/.md/.txt)

## OBSERVED (CUDA)

| Metric | A | B13 | C (beta=1.0) |
|---|---|---|---|
| outcome pass | 0.183 | 0.900 | 0.900 |
| verifier calls (mean) | — | 4.483 | 4.483 |
| distinct programs/task | — | 7.25 | 7.25 |
| family 10 support | — | 0.0 | 0.0 (structural, frozen grammar) |

Gates: G0 identity TRUE · G1 pass TRUE · G2 no-regression TRUE · G3 cost TRUE · G4 diversity TRUE. Paired C vs B13: both 54, C_only 0, B13_only 0, neither 6 → McNemar p=1.0. DB status `ok` (explicit probe, never silent). Cache VRAM 0.047 MiB (64×384 FP16).

## Verdict: `RETRIEVAL_NO_EFFECT` — and the mechanism IS order-invariant

The no-effect is **mathematically necessary**, not a measurement failure:

```
bias_score(c) = energy_score(c) * (1 + beta * sim)      # sim = ONE value per task
energy_score = 0.5 * probs[rule_id]                     # use_energy=False
=> ranking = probs * (1 + beta * sim) ∝ probs           # sim constant → order unchanged
```

One query signature → one top engram → one scalar sim → a **uniform multiplier**. A uniform multiplier cannot re-rank. C's order is byte-identical to B13's for beta>0, which the telemetry confirms (calls and distinct exactly equal).

**Extracted epiplexity:** a single-query engram bias is order-invariant by construction; retrieval can only change candidate ranking if the similarity is **candidate-specific** (per-rule or per-code engram match). The upload's own concept (inject top sim into prompt channel 0) would also be a uniform scalar unless the prompt wave itself is modified per candidate — an unverified claim on a different substrate.

## Deficit dispositions (supplied artifact)

| Claim | Disposition |
|---|---|
| 500K×65,536 ℤ₂₅₆ in 8–12 GB | FALSIFIED (4-bit=16.4 GB, 8-bit=32.8 GB; ℤ₂₅₆ is 8-bit) |
| "O(1) memory lookups" | FALSIFIED (exact sim = O(N·D)) |
| 32 MB cache | CONDITIONAL (256×65,536 FP16 hot subset only) |
| "SMC particle loop in GPU registers" | FALSIFIED (no SMC loop in live eval; upload simulates with randn — mock) |
| sync_timescaledb "silent JSONL fallback" | FALSIFIED (live module raises; JSONL is success export, requires ZONE_C_ENV=prod) |
| A=λA+ψψᵀ "low-rank" per particle | BLOCKED_MISSING_PREMISE (D×D infeasible; needs U∈ℝ^{D×r}; TITANs-equivalence unverified) |
| 65,536-D engram bank | BLOCKED_MISSING_PREMISE (live eval family is 384-D; 65,536-D = third family, Triad rule) |
| R-EDMD fast weights | BOUNDED_IMPLEMENTABLE → v0.6.1-dev (factorized U[r,N], default OFF, identity gate, per-task reset) |
| Heterogeneous sub-swarms | BOUNDED_IMPLEMENTABLE → v0.6.2-dev (PartitionOrder by arity + arg rotation, default OFF) |

## Implemented (all default OFF; baseline byte-identical)

`zone_c_bridge_v060.py`: ZoneCHotCache (live 384-D family), NullZoneCAdapter, ZoneCEngramBias (beta=0 identity), PersistenceStatus (explicit, fail-closed flag), FastWeightRuleMemory (U∈ℝ^{r×N}, λ=0.95, identity at eta=0/reset), PartitionOrder (arity sub-swarms, arg-rotation, cover-each-rule-once). Contracts C1–C9 ALL PASS. Commits `52bf68a`, `fc41feb` (verdict-logic fix: zero-discordance = NO_EFFECT, not REGRESSION).

## Next (pre-registered, separate carriers)

- **v0.6.0.1-dev:** candidate-specific bias (per-rule engram sim) — the actual retrieval hypothesis test; fresh disposable split.
- **v0.6.1-dev:** fast-weight arm (factorized, default OFF) — isolated dev efficacy.
- **v0.6.2-dev:** partition arm — isolated dev efficacy.
- **v0.5.5:** rule-10 semantic fix ONLY (`len({a})`), semantic per-family closure contract (min family support 1.0), NEW single-use heldout seal. NOT bundled with any v0.6.x mechanism.

No heldout was created, consumed, or replayed this cycle. `a09bf275...` remains sealed single-use.
