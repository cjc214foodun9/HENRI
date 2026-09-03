# VLA Grounding Synthesis — Metric Ingress, Causal EFE, Hopfield Egress

**Directive:** `HENRI-DIR-2026-09-BUNDLE-VLA-GROUNDING`
**Source:** `G:\My Drive\HENRI_Inbox\Project_HENRI___henri-bundle_Directive_-_VLA_Ingress__Egress__and_Language_Substrate.md`
**Source hash:** SHA-256 `a4a59cfa019bd67accd6b63a2952474f0fc24cede3c84b3ad4744876f93647ef`, 12,816 bytes (authenticated 2026-09-03)
**Base commit:** `b31f873` (worktree `vla-grounded`, branch `feat/vla-grounded-ml-substrate`)
**Status:** CPU contract suite green (25 passed, 1 CUDA test skipped locally); CUDA gates pending remote execution; causal planner consumer wired behind `HENRI_CAUSAL_PLANNER=1`.

---

## 1. Forensic reconciliation — directive premise claims vs live tree @ `b31f873`

| # | Directive claim | Live-tree finding | Disposition |
|---|---|---|---|
| 1 | `arc_spatial_basis.py` implements global mean-pooling causing G1 AUC collapse (0.7768 < 0.8500) | File is a 1,596-B pure resolver (`resolve_spatial_basis`), zero pooling code; production default = `("incommensurate", True)`; G1 ACCEPTED (module docstring; `henri_vision_encoder.py` encode_grid :94 — per-cell phase superposition, no mean pooling; contract `test_arc_spatial_basis.py` pins defaults) | `STALE` / `FALSIFIED_AT_HEAD` — the 0.7768 AUC is from the pre-7.8 audit doc (`4c88bdcc…`), superseded by G1 ACCEPTED; doc's number absent from live tree |
| 2 | `henri_decoder.py` linear heads (65536→2048→32000) are untrained random noise | Heads exist but are part of the legacy unbinder; checkpoint policy: `required` at d_model=65536, `disabled` below; K2/U2 verdict `BLOCKED_SEMANTIC_CAPACITY` (sealed 2026-08-25); fail-closed `DecoderEgressFailClosedError` live | `CONFIRMED` as a legacy-surface finding, but egress already routed through fail-closed markers + Hopfield snap; new head ≠ sanctioned path |
| 3 | Text codec = random-ring Z_256, dot ≈ 0.0039, zero semantics | Confirmed by sealed run20/codec-geometry control (commit `40ab0b0`); structured codec `qfhrr_structured_codec.py` exists but `FALSIFIED_AT_SCALE` (run21) | `CONFIRMED` — remains the core language-substrate gap; frozen-anchor projection is the bounded mitigation attempted here |
| 4 | EFE planner is a solipsism loop (self-consistency, no Δν) | `efe_planner.py` has `external_outcome_efe` (default-OFF), `external_task_store` (Hopfield), `observe_external_outcome` :451; runner wiring at `production_arc_run.py:177/626` | `STALE` — exteroceptive coupling EXISTS default-OFF (`EXTERNAL_OUTCOME_EFE=1`); new planner module is a bounded clean-room variant, not the first implementation |
| 5 | Entropy starvation H = 1.6807 < 1.70 → `BLOCKED_ENTROPY_GATE` | Sealed CAP12 `c44a00c` — correct | `CONFIRMED` (sealed record, unchanged) |

## 2. Literature anchors (verified / unverified)

- **Verified (arXiv API, hashed responses 2026-09-03):** `2506.21734v3` — Hierarchical Reasoning Model (Sapient Intelligence).
- **In-tree theoretical anchor:** Modern Hopfield continuous energy (Ramsauer et al., cited in `hopfield_cleanup.py` docstring); egress already implements `softmax(β⟨Ψ, M⟩)` snapping at β = 8.0.
- **Unverifiable as cited:** directive's `arXiv:2410.xxxxx` (Attention as Binding) and `arXiv:2203.xxxxx` (Disentanglement with HRR) are placeholders; the specific probed IDs `2008.02217`, `2506.09985` returned no API entries. No mechanistic claim in this doc is anchored to an unverified ID.
- **Zero-pretraining invariant respected:** no pre-trained backbone is loaded, downloaded, or fine-tuned anywhere in this branch. The semantic anchor is an optional frozen random-orthogonal projection (structure, not knowledge) — labeled as such; it is NOT a knowledge substrate.

## 3. Deliverables on branch `feat/vla-grounded-ml-substrate`

| Artifact | Purpose | Notes |
|---|---|---|
| `HENRI V2/henri_metric_ingress.py` | Patch-structured, shift-sensitive complex-phase visual ingress; optional semantic anchor projector (zero-trainable, QR) | No global mean pooling. Color phase inside the frequency ramp (defect caught pre-commit: color must not factor out). Sparse grids: empty patches contribute zero bands; fail-closed only on fully-empty grids |
| `HENRI V2/henri_causal_planner.py` | `BoundedExteroceptiveEFEPlanner`: G(a) = β_prag·d_goal + λ_epis·S_RMS − γ_val·E[Δν]; rolling k=5 failure trace → retroactive ν=−1 + anisotropic noise injection | Consumes caller-supplied predictions; does NOT reimplement transition dynamics; lazy import and default-OFF consumer flag `HENRI_CAUSAL_PLANNER` |
| `HENRI V2/henri_hopfield_egress.py` | Canonical-codebook validator layer over the LIVE `ContinuousHopfieldCleanup` core (β=8.0) | Adds fail-closed syntax rejection (`EgressSyntaxRejectedError`), canonical-id allowlist, `noise_floor_sigma = ε/√D` helper. Hopfield core itself was ALREADY IMPLEMENTED (`henri_egress.py`) |
| `HENRI V2/darwinian_phase_swarm.py` | Approved live consumer wiring | Flagged branch feeds live `EFEPlanner.transition` predictions into the bounded causal scorer; default path calls the existing planner unchanged |
| `HENRI V2/production_arc_run.py` | External-outcome wiring | Forwards only observed `task_progressed` deltas to the causal consumer; frame change is not used as a score proxy |
| `HENRI V2/tests/contract/test_vla_grounded_components.py` | Gate matrix plus consumer/device contracts | 25 passed, 1 CUDA boundary test skipped locally |
| `docs/spec/vla_grounding_synthesis.md` | This document | |

## 4. Gate matrix status (directive §4)

| Gate | Status | Evidence |
|---|---|---|
| G-INGRESS (AUC_contact ≥ 0.8800) | PASS (CPU proxy, D=512) | 7-distractor rank proxy ≥ 0.88; suite `test_contact_auc_proxy` + anti-collapse color/shape guards |
| G-SOLVER (≤ 2.0 ms, ‖K‖₂ ≤ 1.0) | `BLOCKED` — remote CUDA gate | Live Koopman fit (`BlockRidgeKoopmanFit`) untouched; no `APPROVE_REMOTE_RUN`; A2 measured the screen stage dominant; a solver-timing claim needs the sealed CUDA harness |
| G-ENTROPY (H ≥ 1.70, N(a) ≥ 15) | `BLOCKED` — live-run gate | CAP12 `c44a00c` sealed; no remote run authorized |
| G-VALENCE (Δν over self-consistency) | PASS (CPU proxy) | `test_positive_delta_lowers_g`, `test_select_action_argmin`, failure-trace tests |
| G-EGRESS (20/20 recovery, SNR ≥ 20 dB) | PASS (CPU proxy, D=512) | `test_20_of_20_recovery_under_floor_noise`; syntax-rejection tests |

## 5. Honest limits (evidence labels)

- All module tests are `OBSERVED` CPU-proxy results, NOT CUDA verification and NOT external task outcomes.
- Ingress, validator egress, and the causal scorer are not equivalent to a complete VLA or language system. Only the causal scorer has an approved production consumer, and it remains default-OFF.
- The semantic anchor supplies orthogonal geometry, not world knowledge — it does not close the AAII v4.1.1 knowledge gap.
- The Hopfield egress validator does not add capacity; U4 `BLOCKED_EGRESS_CAPACITY` and K2/U2 `BLOCKED_SEMANTIC_CAPACITY` verdicts stand.
- Sealed records unchanged: KG5′ instrument at `b31f873` still requires `APPROVE_REMOTE_RUN`; carrier-k3 worktree untouched.

## 6. Execution receipt

Commit (local-only, no push): `3cacad6` on `feat/vla-grounded-ml-substrate` at base `b31f873`.
Files: the five artifacts above. Push and production wiring remain approval-gated.
