# System-1 v0.6.1-dev — Fast-Weight Epistemic Memory: FASTWEIGHT_NO_EFFECT

**Date:** 2026-08-24 (CUDA 5090, vast-5090, PID 402174)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen)
**Split:** dev7b_v061 (DISPOSABLE, seed 71728, n=60, sha `338cbda6`)
**Arms (matched budget 64 / beam 64):** A token beam · B13 uniform CEGIS-first · D + fast-weight failure-downweighted admission

## OBSERVED (CUDA)

| Metric | A | B13 | D (rank=8, eta=0.5, λ=0.95, per-task reset) |
|---|---|---|---|
| outcome pass | 0.217 | 0.917 | 0.917 |
| verifier calls (mean) | — | 5.033 | 5.033 |
| distinct programs/task | — | 7.42 | 7.42 |
| family 10 | — | 0.0 | 0.0 (structural, frozen grammar) |

Paired D vs B13: both 55, D_only 0, B13_only 0, neither 5 → McNemar p=1.0. Gates G0–G4 all TRUE. Verdict: **`FASTWEIGHT_NO_EFFECT`**.

## Mechanism (why zero discordance)

`U ∈ ℝ^{r×N}` factorized failure memory, `adjusted_probs` with `rule_ids` alignment (arity-filtered pools). Reordering only acts AFTER a verifier failure, and it reorders among NON-admitted candidates; the correct rule's probability is untouched (it never fails). In the uniform CEGIS-first pool the correct rule is already admitted at mean 5.0 calls — the downweighted trailing candidates never affect admission. At 0.917 baseline, the effect is structurally invisible on 60 tasks.

Honest statement: failure-downweighting changes order only among already-failing candidates; it cannot move the first-pass candidate earlier when the prior already ranks it first. The mechanism is not harmful; it is inert at this operating point.

## Integrity

- Stale-dependency defect caught by remote sha check: first launch (PID 401884) crashed on `adjusted_probs(rule_ids=...)` because `zone_c_bridge_v060.py` on the remote predated the fix. Split `dev7_v061` (sha `392ce03e3b35`) was partially exposed → **quarantined into the consumed-digest guard**; fresh seed 71728 + tag `dev7b_v061` sealed; relaunched. No replay.
- No heldout created or consumed. `a09bf275` untouched.

## Next (per separation map)

v0.6.2 partition (executed, see SYSTEM1_V062_VERDICT.md). v0.5.5 rule-10 fix remains separate (semantic per-family closure contract, NEW seal). Fast-weight requires a prior that does NOT already rank the correct rule first to have any measurable window — not promotable at this baseline.
