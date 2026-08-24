# System-1 Stage-0c-rev3 — r=16 Reduced-Koopman Spectral Evaluation (2026-08-24)

**Verdict: CONTRACT_FAILED (first failing gate C8)** · Reference 3 (gpt-5.6-sol) binding
Prior rev `IDENTIFIABILITY_BLOCKED` + rev2 `CONTRACT_FAILED` preserved. No verified verdict emitted.

## Upload (proposal, audited)
`Stage-0c-rev3_Evaluation_and_Authorization.md`, 739 B, SHA-256 `99d28342…`. 15-line diagram:
top-16 share > 92%, κ16 = 5.49, orthogonal residual < 8%, floor < 0.065, SSR target < 0.350;
user protocol: SSR_eval ≤ 0.40, ε_rollout,5 ≤ 0.35. C-text/definitions authored + sealed.

## Pre-seal probe (OBSERVED, sha `a34b50f9…`, read-only)
- top-16 share 0.9341/0.9414 → PASS; κ16 5.4948/5.9771 → PASS (≤10).
- **"absolute error floor < 0.065" FALSIFIED** — 0.065 is the variance fraction (1−0.9341); the
  true full-space floors on Y-target are 0.335/0.367. Sealed metric = projected coefficient-space.
- Fresh eval corpus: 220 records, seeds 2101–3010, manifest `f0c9a7624f26bf70…`,
  raw-obs overlap vs calib = 0 (OBSERVED).

## Contract: `vla_stage0c_rev3_contract.md` (sha `87b28acb…`, sealed BEFORE K; prereg `74bca9b3…`)

## Results (OBSERVED; runs 1+2; C11 determinism PASS)
| Gate | Result |
|---|---|
| C7 κ16≤10, top16≥0.92 | PASS (5.49/5.98; 0.934/0.941) |
| C8 calib projected ε≤0.05 | **FAIL** — a0 0.1260, a1 0.1286 |
| C9 SSR_eval≤0.40 (fresh eval) | PASS — agg 0.3688 (a0 0.3893, a1 0.3484); eps_eval 0.187/0.175 vs persistence 0.481/0.502; calib-mean 0.824/0.841 |
| C10 ρ≤1.05 AND rollout≤0.35 | FAIL — ρ 1.0095/0.9555 (PASS); rollout 0.555 agg > 0.35 (0.572/0.541) |
| C1–C6 | PASS (bypass, zero-trainable, npz sha, sens 100%, full rank, PR 7.03/6.69) |
| C12 baselines | persistence 0.481/0.502; calib-mean 0.824/0.841 |

## Disclosure (no rerun; verdict unaffected)
- `c4_sphere_max_err = 3.0` is a RESHAPE DEFECT in the rev3 diagnostic: it norms the flat 6144-D
  vector (per-slot norm √16 = 4 → |4−1| = 3) instead of per-slot 384-D. C4 is NOT in the sealed
  rev3 gate chain. The same encoder was verified per-slot in rev2 (sphere max err 5.96e-08).
  Kept as-is to preserve the sealed evidence chain; cited rev2 value is the correct geometry.

## Interpretation
- The upload's headline SSR gate PASSED on genuinely fresh disjoint episodes (2.6–2.9× better
  than persistence in projected space) — first positive transfer signal in the Stage-0c lineage.
- The upload's rollout gate FAILED (0.555) — corpus #26 (INFERRED: r=16 > PR~7 → 5-step rollout
  ≤0.35 "physically impossible") CONFIRMED. ρ(a0)=1.0095 > 1 compounds.
- Sealed C8 (0.05 interpolation) fails at PR~7 exactly as the over-parameterization prediction.
- Conclusion: relative skill is real; absolute calibration and multi-step stability are not.

## Artifacts
- Contract `87b28acb…`; probe `a34b50f9…`; telemetry `6c6df872…`; operators `d6953453…`
- Eval corpus manifest `f0c9a762…` (220 recs); governance prereg `74bca9b3…` + outcome `…`

## Boundaries
- CartPole dynamics result only. **VLA gate 0/12. AAII v4.1.1 0/9 BLOCKED.** No SOTA claim.

## Next options (require decision)
- (A) Stage-0c-rev4: r=8 with the fresh eval corpus + SSR/rollout gates only (drop the 0.05
  calibration gate; C8's PR~7 floor makes it structurally unreachable), pre-registered.
- (B) Persistence-anchored relative-only gates on a NEW carrier (never relabel this audit).
- (C) Richer substrate / different environment.
- (D) Hold at `CONTRACT_FAILED_ACCEPTED` (rev3).
