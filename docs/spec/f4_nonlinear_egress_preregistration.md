# Carrier F4: Non-Linear Context-Conditioned Egress — Pre-Registration

**Document ID:** HENRI-SPEC-2026-08-F4-NONLINEAR-EGRESS
**Branch:** carrier/f3 (spec accumulates with prior carrier docs)
**Status:** PRE-REGISTERED (sealed; implementation REQUIRES_APPROVAL)
**Parent verdict (ratified):** `F3_GATES_VERDICT=K1_KILLED` — governance event `8c47bf5c`, audit hash `b4601cd2…`, ledger depth 997
**Source directive:** `Project_HENRI_F3_Gauntlet_Post-Mortem___Non-Linear_Egress_Reform_Directive.md` — SHA-256 `a2b2640a85b47261b0800b4deed2d87bc019168285cf759b951837f02b3dcad7`, 15,189 B, ID `HENRI-DIR-2026-08-F3-GAUNTLET-POSTMORTEM-EGRESS-REFORM`

---

## 0. Executive determination

The F3 post-mortem directive is RATIFIED with one disclosed correction:

- **RATIFIED:** static linear codebook egress is falsified on the validated F3 v2 bank (G1 macro P@1 0.178, min-fold 0.090 vs gates ≥0.99/≥0.95; G4 margin +0.0965 PASS). The corpus is a valid instrument (H=1.7992 nats, CV_diff 1.0207, all 12 envs ≥100 rows, all actions ≥30 rows).
- **DISCLOSED PROSE DRIFT (§2.1 of the directive):** the fold table names `cn04`, `dc22`, `lf52` — none of which exist in the 12-env bank — and per-fold P@1 values (0.2140/0.1820/0.0900/0.2260) that do not match the sealed receipt (fold0 actual P@1 0.2418, marginal 0.0; folds per `f3_gates_receipt.json`). The verdict and architecture sections are unaffected; this pre-registration binds to the SEALED receipt values, never the prose table.
- **DISCLOSED G2 nuance:** per-fold per-action `n_test=0` with `coverage_limited: true` for ACTION5/6/7 in some folds (sealed receipt). The F4 gate below therefore requires `n_test ≥ 10` to score an action, per F2 convention.

## 1. Falsified mechanism (do not re-propose)

Static dual-ridge projection `M = YᵀΨ(ΨᵀΨ + λI)⁻¹`, `M ∈ C^{|A|×D}` applied identically across environments: **FALSIFIED** at D=65,536 by the sealed F3 gauntlet. Any carrier that re-proposes an environment-agnostic linear codebook with no new evidence is rejected without a run.

## 2. F4 mechanism (three tiers)

Inputs: `Ψ ∈ R^{1×65536}` (fp16 bank rows, cast fp32 at head input), task identity `env_id`, per-env demo prefix rows `{(Ψ_i, y_i)}` (defined §4.3).

### Tier 1 — Task-functor phase pre-inversion
`Ψ_unbound = normalize(Ψ ⊛ W_task†) ∈ S^{D-1}` (circular-convolution unbinding in the qFHRR phase domain; `W_task ∈ Z_256^D` compiled per-env from the env's demo prefix via `HolographicTaskFunctorCompiler.compile_functor` / `progressive_semantic_grounding_engine.compile_functor_wave` — live symbols, henri_decoder/zone_c path).

### Tier 2 — Deep non-linear compression head
```
h1 = GELU(LN(W1 · Ψ_unbound))     W1 ∈ R^{2048×65536}
h2 = GELU(LN(W2 · h1))            W2 ∈ R^{512×2048}
z  = W3 · h2                      W3 ∈ R^{7×512}
a* = argmax_k z_k
```
Params ≈ 135.3M (W1 134.2M + W2 1.05M + W3 3.6k); fp32 weights ≈ 541 MB, fp16 activations. Trainable: W1,W2,W3 at train time (folds); **Tier 3 adapts W3 only**.

### Tier 3 — In-situ test-time adaptation (SGLD, W3 only)
On the heldout env's demo prefix only (≤20 rows):
```
W3 ← W3 − η ∇_{W3} L + sqrt(2·T(t)·dt)·ξ,   L = CE(z, y_demo) + 1e-4·||W3||²
T(t) = T0·(1 + 0.05·t)^{−0.55}, T0 = 0.5, dt = 1.0, η = 1e-3, steps = 3
```
Noise ξ ~ N(0,1) unit-normalized per step. W1/W2 frozen during Tier 3. `adapt_in_context_sgld_wave` (henri_decoder.py:209) is the reference protocol; the action-head variant replaces soft wave targets with one-hot action labels (logits are not waves; no Sagnac term).

## 3. Pre-registered gates (acceptance / rejection)

| Gate | Metric | Threshold | Notes |
|---|---|---|---|
| G1 | Macro P@1 across heldout rows | ≥ 0.99 | same convention as F3 |
| G1b | Min-fold P@1 | ≥ 0.95 | 4 grouped folds |
| G2 | Per-action P@1, actions with n_test ≥ 10 | min ≥ 0.80 | coverage-limited actions excluded, reported separately |
| G3 | Payload-format valid (ACTION6 coordinate payloads) | ≥ 0.99 | requires schema v2 payload capture; if bank lacks payloads → `BLOCKED_NO_PAYLOAD_IN_BANK`, not PASS |
| G4 | Margin vs train-marginal predictor | ≥ +0.05 | per-fold, macro |
| G5 | **Paired bootstrap CI (lb > 0)**: A(full) vs D(linear+Wtask, matched protocol) | lb > 0 on per-env P@1 deltas | proves nonlinearity gain under identical information |
| G6 | Paired bootstrap CI: A vs C(no Tier 1) | lb > 0 | proves task-conditioning gain |
| G7 | Paired bootstrap CI: A vs B(no Tier 3) | lb > 0 | proves in-situ adaptation gain |

**Pre-registered verdicts:** `F4_EGRESS_PROMOTED` (G1+G1b+G2+G3+G4 and G5+G6+G7 all pass) · `FALSIFIED_NO_EXTERNAL_GAIN` (any absolute gate fails, or any control ties A within CI) · `BLOCKED_INFRASTRUCTURE` (nonzero arm exit / missing artifact) · `BLOCKED_TARGET_LEAKAGE` (provenance scan fails) · `CONDITIONAL_REUSED_EVAL` (if fresh-split discipline is violated — forbidden by default).

## 4. Data path and split policy (anti-leakage)

### 4.1 Corpus
`trajectories_production_run_f3v2.npz` — `psi [1536,65536] f16`, `next_wave [1536,65536] f16`, `actions_onehot [1536,7] u8`, `action_names (7,)`. Hash pinned in `f3_split_seal.json` (`npz_sha256`). Per-env counts: ar25 134, bp35 118, cd82 100, ft09 150, g50t 130, ka59 100, lp85 150, sb26 122, sc25 104, sk48 150, tr87 128, wa30 150 (Σ=1536).

### 4.2 Fold policy — FRESH SEAL REQUIRED (F3 split is consumed)
The F3 split (seed 20260829, fold manifest `30504659…`, `single_use=true`) is **CONSUMED** and will NOT be reused. F4 seals a NEW grouped 4-fold split: env-disjoint (3 heldout envs × 4 folds), NEW seed (default `20260830`), `single_use=true`, full SHA-256 receipts, generation-only process. Any F4 result produced on the F3 split is `CONDITIONAL_REUSED_EVAL` and fails the carrier.

### 4.3 Demo prefix (test-time task conditioning)
Per env, rows are ordered by capture time; the FIRST `k=20` rows form the demo prefix (max 20, floor min(20, n_env/5)); the remaining rows are evaluation rows. Rules:
- Train-fold envs: demo prefix participates in head training (it is train data).
- Heldout-fold envs: demo prefix is used ONLY for Tier 1 W_task compilation and Tier 3 SGLD adaptation at eval time. It NEVER enters head training and is EXCLUDED from evaluation rows.
- Static provenance scan (pre-run): assert zero heldout-eval-row leaks into W_task compilation, head training, or hyperparameter selection.

### 4.4 Hyperparameters (fixed, no tuning on heldout)
Seed 20260830; AdamW lr 1e-3, β=(0.9,0.999), weight decay 1e-4; batch 64; epochs ≤ 20 with early stop on TRAIN-fold validation envs only; LN eps 1e-5; GELU exact; fp32 head; ridge λ=1e-3 (Tier-1 compile); W1/W2 init Kaiming, W3 zero-init.

## 5. Arms (multi-arm kill matrix)

| Arm | Tier 1 | Tier 2 | Tier 3 | Purpose |
|---|---|---|---|---|
| A | ON | MLP | SGLD(3) | full F4 |
| B | ON | MLP | frozen | isolates adaptation |
| C | OFF | MLP | SGLD(3) | isolates task conditioning |
| D | ON | linear dual-ridge | — | matched-protocol linear control (same W_task, same demos) |
| E | — | train-marginal predictor | — | gate baseline |

Any nonzero arm exit → `BLOCKED_INFRASTRUCTURE` for the whole run (no per-arm science).

## 6. Cheapest kill experiments (pre-registered, run BEFORE full gates)

1. **Nonlinearity sanity:** synthetic 2-class XOR-like wave problem — MLP must beat linear ridge; failure = harness defect, fix harness, no verdict.
2. **Tier-1 engagement:** `cos(Ψ, Ψ_unbound) < 0.99` on heldout-env rows and action-separation increase (mean intra-action cos ↑) — else `FALSIFIED_NO_ENGAGEMENT` for Tier 1.
3. **Tier-3 engagement:** after 3 SGLD steps on a heldout env demo prefix, `‖ΔW3‖ > 1e-6` AND CE(demo prefix) descends — else Tier 3 dead.
4. **Default-OFF differential:** with `HENRI_F4_EGRESS` unset, production runner output byte-identical (wiring stage).
5. **Leakage scan:** static provenance audit of every tensor entering Ψ_task / training (per §4.3). Failure → `BLOCKED_TARGET_LEAKAGE`, stop.

## 7. Resources and constraints

- Remote CUDA only: Vast RTX 5090 (32 GB); MLP fp32 ≈ 541 MB — fits.
- Local CPU tests: contract/shape only, never verdicts.
- Zero edits to `production_arc_run.py` (3163-line runner invariant); F4 lives at orchestration layer + new module `f4_nonlinear_egress_head.py` (default-OFF flag `HENRI_F4_EGRESS=1`).
- Dense `[65536,2048]` matmul is allowed (single head, fp16/fp32); NO dense `[D,D]` tensors anywhere else.

## 8. Verification plan (TDD, then remote)

1. Contract tests first (RED): head shapes, LN/GELU path, W_task compile from demo prefix, SGLD W3-only update, fold loader provenance scan, gate arithmetic. GREEN 21+ tests.
2. Exact-SHA detached worktree on Vast; SHA-identical to pushed commit; overlay preflight.
3. Kill experiments 1–3 on CUDA (bounded smoke, disposable seeds — NOT the sealed split).
4. Seal fresh F4 split (generation-only, no checkpoint load).
5. Full arms A–E on the sealed split; aggregate per-fold receipts; bootstrap CIs (10k resamples, per-env paired deltas).
6. Verdict + governance seal; main untouched; `HENRI_F4_EGRESS=0` until promotion approval.

## 9. Governance commitments

- This document SHA-256 sealed via `F4_PREREG_SEALED` audit event (child of `8c47bf5c`).
- Any pre-seal amendment changes the SHA: recompute, record both, re-seal — never edit silently.
- Promotion to main requires: all gates pass + explicit user approval (load-bearing change).
- Negative result seals as `FALSIFIED_NO_EXTERNAL_GAIN` and keeps the carrier default-OFF — negatives are governance wins.

## 10. Next actions

1. [THIS DOC] seal + commit + governance event.
2. USER APPROVAL GATE: approve implementation (TDD harness + remote CUDA arms).
3. Implement `f4_nonlinear_egress_head.py` + contract tests (RED→GREEN).
4. Remote kill experiments, fresh split seal, full arms, verdict.
