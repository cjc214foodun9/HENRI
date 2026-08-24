# System-1 Stage-0c-rev2 — Reduced-Koopman Spectral Evaluation (2026-08-24)

**Verdict: CONTRACT_FAILED** · Reference 3 (gpt-5.6-sol) binding
No `IDENTIFIABILITY_VERIFIED` sealed. Prior Stage-0c-rev `IDENTIFIABILITY_BLOCKED` preserved (not relabeled).

## Upload (proposal artifact, audited)
- `Stage-0c-rev_Spectral_Evaluation___Stage-0c-rev2_Authorization.md`, 562 B, SHA-256 `99b3b828…`
- 13-line diagram: PR=7.03, κ16=5.49, top-8 = 88.4% (claim), K_r for r∈{4,8}, κ8<3.0, ε_EDMD≤0.05
- C1–C12 text, K construction, baselines, stability, selection rule NOT specified → authored + sealed

## Pre-seal corrections (disclosed; old sha `152130b3…` → final `dc3d3d65…`)
1. Upload "κ8 < 3.0" FALSIFIED by read-only probe (a0 2.951, a1 3.085) → gate κ8 ≤ 10.0.
2. Upload "top-8 = 88.4%" FALSIFIED (a0 0.796, a1 0.789) → gate ≥ 0.75.
3. Upload "ε_EDMD ≤ 0.05" internally contradictory with its own 88.4% claim: r=8 FULL-SPACE floor
   0.51–0.68 (measured) → full-space ε≤0.05 mathematically infeasible. Sealed metric = PROJECTED
   (coefficient-space) normalized Frobenius; full-space errors diagnostic.
4. Upload selection `r* = argmin ε_eval(r)` REJECTED (evaluation-set model selection); sealed rule:
   r* = argmin mean_a ε_proj_calib(a,r), tie → r=4; evaluation scored ONCE on r*.
- Governance: PREREG `7921f420…` sealed BEFORE any K construction.

## Contracts C1–C12 — results (OBSERVED, runs 1+2)
| Gate | Result |
|---|---|
| C1 default-OFF bypass | PASS (byte-identical) |
| C2 zero trainable state | PASS (class-source AST scan, no hits) |
| C3 npz sha assert | PASS (`766e607a…` full match) |
| C4 sphere geometry | PASS (max err 5.96e-08 ≤ 1e-6) |
| C5 sensitivity | PASS (181 distinct calib obs; 100% pairs L2>1e-3; min 0.280; 0 collisions) |
| C6 full rank + PR>4 | PASS (r(>1e-6)=102/102, 69/69; PR 7.03/6.69) |
| C7 κ8≤10.0, top8≥0.75 | PASS (a0 2.951/0.796, a1 3.085/0.789) |
| C8 calib projected ε≤0.05 | **FAIL** — a0_r4 0.177, a0_r8 0.167, a1_r4 0.181, a1_r8 0.145 |
| C9 eval projected ε≤0.05 + ratio≤0.95 | **FAIL** — eval a0 0.211, a1 0.216 (r*=8); persistence 0.447/0.447; ratio 0.471/0.483 (relative PASS, absolute FAIL) |
| C10 ρ≤1.05 + coeff rollout≤0.15 | **FAIL** — ρ a0 0.949, a1 0.927 (PASS); rollout a0 0.733, a1 1.052 (FAIL) |
| C11 determinism | PASS — telemetry + operators byte-identical across 2 separate processes |
| C12 baselines reported | persistence 0.447/0.447; calib-mean 0.876/0.869 |

- r* = 8 (calibration selection: mean ε_proj r4 0.1793, r8 0.1560).

## Verdict chain (sealed order)
CONTRACT_FAILED — C8 is the first failing sealed gate; C9 and C10 also fail independently.

## Interpretation
- Upload claim "ε_EDMD ≤ 0.05 (Unblocked)" FALSIFIED by the actual fit (projected calib 0.145–0.181).
- Reduced EDMD captures real dynamics vs baselines (2.1× improvement over persistence in projected
  one-step error; calib-mean 0.87 far worse) but fails every pre-registered absolute gate; rollout
  error 0.73/1.05 shows one-step fit does not imply multi-step stability — regression engagement,
  not verified learned dynamics (Reference 3 pitfall).

## Artifacts
- Contract `vla_stage0c_rev2_contract.md` (`dc3d3d65…`); script `vla_stage0c_rev2_spectral.py`
- Telemetry `vla_stage0c_rev2_telemetry.json` (sha `80d4671cfcb691e0642f8b59…`)
- Operators `vla_stage0c_rev2_operators.npz` (sha `e1c6f83c…`)
- Governance: prereg `7921f420…` + outcome `03c2a91d…`

## Boundaries
- VLA gate **0/12**. AAII v4.1.1 **0/9 BLOCKED** — no causal path from this spectral carrier to any
  index component; no SOTA claim.

## Next options (require decision)
- (A) Stage-0c-rev3: r=16 on the same frozen encoder (N_a≥4r: 64≤102, 64≤69 holds; κ16 5.5–6.0
  measured) with pre-registered relative-to-persistence gates + fresh protocol.
- (B) Persistence-anchored relative-only gates under a NEW pre-registration (never relabel this audit).
- (C) Richer substrate / different environment.
- (D) Hold at `CONTRACT_FAILED_ACCEPTED`.
