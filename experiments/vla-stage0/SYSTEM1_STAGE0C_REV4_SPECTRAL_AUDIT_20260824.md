# System-1 Stage-0c-rev4 — r=8 Contractive Spectral Evaluation (2026-08-24)

**Verdict: CONTRACT_FAILED (first failing gate C9)** · Reference 3 binding
Prior rev/rev2/rev3 verdicts preserved. No verified verdict emitted.

## Upload (proposal, audited)
`Stage-0c-rev4_Architecture___Pre-Registration_Protocol.md`, 612 B, SHA-256
`458d07761b3a7741badfee3473fcfd5155ae96d441c001c38f6b26b603ad0efd`. 16-line diagram:
UNSEEN x(t) → frozen RFF [6144D] → V8 (κ8 2.95, VarShare 79.6%) → EDMD + spectral projection
ρ(K8) ≤ 1.0000 → SSR_eval ≤ 0.40 ("Passed: SSR = 0.3688") + SSR_rollout ≤ 0.80 ("Unblocked by ρ ≤ 1.0").

## Pre-seal
- Contract `cfa2a1d5…` sealed BEFORE K (prereg `a0c6b20d…`); r=8 fixed; per-action V8/K8;
  contraction rule: radial clamp if ρ>1.0, identity arm if ρ≤1.0, cond(U)>1e8 → BLOCKED_NUMERICAL.
- Consult #29 (INFERRED): r=8≈PR~7 defensible; SSR_rollout5≤0.80 vs persistence-5 defensible;
  clamping pathologies documented (cond(U) explosion, complex-pair damping loss, non-normal transients).
- Eval corpus: rev3 220 records (seeds 2101–3010) — labeled `CONDITIONAL_REUSED_EVAL`.

## Results (OBSERVED; runs 1+2; C11 determinism PASS)
| Gate | Result |
|---|---|
| C8 κ8≤10, top8≥0.75, ρ(K̃)≤1.0 | PASS — κ8 2.9507/3.0848; top8 0.7962/0.7894; ρ 0.9489/0.9266; **contraction NEVER fired** (ρ<1 both actions); cond(U) 2.46/2.22; snorm_raw 1.042/1.016 (>1.0 with ρ<1.0 = non-normal transient, corpus #29 documented) |
| **C9 SSR_eval ≤ 0.40** | **FAIL** — agg 0.4233 (a0 0.4558, a1 0.3909); eps_eval 0.1998/0.1718 vs persistence1 0.438/0.440 |
| **C10 SSR_rollout5 ≤ 0.80** | PASS — agg 0.5160 (a0 0.5453, a1 0.4867); eps_roll5 0.6136/0.5566 vs persistence5 1.1252/1.1437 (2.2× better) |
| C1/C3 diagnostics | bypass PASS; npz sha PASS |

## Upload-claim reconciliation
- **"SSR_eval = 0.3688 Passed" is FALSIFIED at r=8** — 0.3688 is the rev3 **r=16** result
  (telemetry `6c6df872…`). At r=8 the same metric = 0.4233 > 0.40. The upload conflated ranks.
- **"Unblocked by ρ ≤ 1.0"** — the contraction premise was vacuous: ρ_raw < 1.0 for both actions,
  so no projection was applied. The 5-step rollout DID pass (0.516) but not because of contraction.
- Rank law measured again: r=16 SSR_eval 0.369 PASS → r=8 0.423 FAIL (lower rank costs one-step
  skill); r=16 rollout 0.555 absolute FAIL → r=8 SSR 0.516 vs weak persistence-5 (relative PASS).

## Interpretation
Relative skill is consistent (one-step and 5-step both beat persistence), but the r=8 carrier
fails its own absolute one-step gate; absolute calibration remains absent across all four
carriers. The upload's premise chain (r=8 fixes the r=16 over-allocation) is FALSIFIED — the
tradeoff is one-step skill at r=16 vs multi-step relative stability at r=8, with no rank meeting
both absolute gates.

## Artifacts
- Contract `cfa2a1d5…`; prereg `a0c6b20d…`; outcome `…`; telemetry `ed162c85…`; ops `e4b2d744…`

## Boundaries
- CartPole dynamics result only. **VLA 0/12. AAII v4.1.1 0/9 BLOCKED.** No SOTA claim.
- Architecture program: separate decision record addresses the roadmap (`6c534e93…`) disposition;
  this carrier does not advance AAII capability.

## Next options (require decision)
- (A) Stage-0c-rev5: r∈{8,16} both measured on ONE fresh eval corpus with relative-only gates
  (persistence-anchored SSR) + fresh disjoint eval (seeds 4010+); never mix ranks across carriers.
- (B) Persistence-anchored relative-only gates on a NEW carrier.
- (C) Abandon the CartPole spectral line; proceed to the semantic-backbone → structural-AST-egress
  carrier (the measured MBPP bottleneck; Reference 3 recommendation).
- (D) Hold at `CONTRACT_FAILED_ACCEPTED` (rev4).
