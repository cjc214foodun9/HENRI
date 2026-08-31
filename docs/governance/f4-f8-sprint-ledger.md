# HENRI F4–F8 Sprint Ledger

**Status:** DRAFT (not yet sealed as a governance event)
**Compiled:** 2026-08-30, from the immutable audit chain
**Source of truth:** `C:\Users\chan\AppData\Local\hermes\audit\henri_audit_chain.jsonl`
**Chain verification:** `henri_audit.py verify` → `chain intact: 1043 records, head fb2535a3a8ae0eb8…` (OBSERVED)
**Evidence class:** every number below is OBSERVED from sealed verdict events (CUDA gauntlets at pinned commits). No prose number was substituted for a receipt value. MoA reference output was excluded: Ref-1/Ref-2 returned empty; Ref-3 emitted fabricated `[called tool:…]` blocks (pre-patch fabrication class, strip filter `3962091b24` live after restart).

---

## 1. Carrier chain map (ledger indices 999–1041)

| Carrier | Prereg | Impl | Amendment | Verdict | Verdict hash |
|---|---|---|---|---|---|
| F4 nonlinear egress | `#5b04d6b5` (999) | `1c0869d7`→`214aba3b` (1003/1004) | `#287529ec` (1002) | **K1_KILLED** | `#392a23ff` (1006) |
| F5 FPB codec | `#cf7328cd` (1011) | `178a45a` (1012) | — | **FALSIFIED_AT_SCALE** | `#5a04c478` (1013) |
| F6 adaptive functor | `#644286e6` (1016) | `#1d9a1d17` (1018) | — | **FALSIFIED_NO_GEOMETRY** | `#8a0fd516` (1019) |
| F7 affine egress | `#c360ad42`→`#757d658a`→`#fe25c6b8` (1022–1025) | `#6a8cf948` (1026) | Appendix C | **FALSIFIED_AT_SCALE** | `#177c30b5` (1027) |
| F8 decodability probe | `#364346fea5` (1032) | `#52a01d6b` (1033) | `#470e6b58` (1039) | **F8_INDETERMINATE / CASE_C** | `#04df0f9f` (1041) |

Chain integrity: each `prev_hash` links to its predecessor; `verify` reports zero mismatches. Ledger head after F8: `fb2535a3…` (idx 1042 = CONTEXT_HANDOFF).

---

## 2. F4 — nonlinear egress head (`carrier/f4`)

- Directive `a2b2640a…`; spec `15e3376e…` → amended `7c85102f…` (A1–A6: seeded-permutation split, Tier3 T0=1e-6, exact params 135,272,455, SGLD noise sign +, dual thin-SVD arm D, kill smoke). Arms A–E; gates G1–G7.
- Impl: `HENRI_F4_EGRESS=1` default-OFF, runner untouched; contracts 16 F4 + 21 F3 = 37 GREEN; fixes committed at `214aba3b`.
- **Verdict `K1_KILLED` (#392a23ff):** g1_macro_p1_A **0.2296** (min fold 0.1414), g2_min_scored_p1 **0.0**, g3 `BLOCKED_NO_PAYLOAD_IN_BANK`, g4_macro_margin_A 0.1437, g5_lb −0.0807, g6_lb −0.0355, g7_lb 0.0. Kill smoke OK. Split seed 20260830, digest `640763c6…`.
- Disposition: *F2 and F4 egress candidates FALSIFIED; evidence-only; `HENRI_F4_EGRESS=0`; main untouched.*

## 3. F5 — Fourier fractional-power-binding codec (`fpb_qfhrr_codec.py`)

- Directive `09c0fccc…` (16,938 B, `HENRI-DIR-2026-08-F4-POSTMORTEM-CODEC-REFORM`; byte-identical duplicate of run21 commit `440f11d`). Prereg `#cf7328cd` spec `f4b1a2c1…`; new mechanism `fourier_domain_fractional_power_binding`; corpus consult `BLOCKED_REPL_ONLY_nlm_cli`.
- Impl `178a45a`: `HENRI_F5_CODEC=1` default-OFF, runner untouched, 53/53 regression, 16 GREEN; calibration homo 1.00000 / rho_wave 0.932 / far −0.001 (OBSERVED CPU).
- **Verdict `FALSIFIED_AT_SCALE` (#5a04c478, seed 20260831):** arms A_fpb **0.4352** / B_run21 0.439 / C_legacy **0.4414** / D_identity 0.1597; G3 0.4352 < 0.8 FAIL; **G4 −0.0062 < 0.5 FAIL**; G5 per-env occlusion 0.0–0.35 confirmed. Kill smoke g2 0.8781 / homo 1.0 / rho 0.9377. Split rule: grouped 4-fold env-disjoint seeded permutation mod.

## 4. F6 — adaptive unitary functor (`carrier/f6-adaptive-functor`)

- Directive `73299bd4…`; prereg `#644286e6` spec `77877ce8…`; gates G1 1e-5 / G2 0.90 / G3 0.75 / G4 +0.30; corrections: circulant spectral NS dense-infeasible, per-env F5 harness reconciliation.
- Impl `#1d9a1d17`: `HENRI_V2/f6_adaptive_functor.py`, wiring `arc_task_functor.py HENRI_F6_FUNCTOR=1` additive default-OFF; C1–C8 8/8 + F5+F6+arc 27/27; corrections spectral-norm W0 normalization, C2/C3/C5 unitary fixtures.
- **Verdict `FALSIFIED_NO_GEOMETRY` (#8a0fd516, commit `752d915`):** arms A_adaptive 0.436 / B_f5control 0.4352 / C_legacy 0.4414 / D_identity 0.1597; G1_ns_err 26.60; G2_recon 0.7763; G3 0.436; **G4_margin_AB 0.0008**. Interpretation (sealed): *real-bank demo relation X→action is not a unitary operator (20 rows → 7 actions); de-occlusion mask inert on quantized waves; armA == armB confirms retraction+mask+snap change nothing.*

## 5. F7 — affine egress (`carrier/f7-affine-egress`)

- Directive `3ea072ea…`; prereg chain `#c360ad42` (PLACEHOLDER defect) → `#757d658a` (corrected, pre-correction sha `a3306c86…`) → `#fe25c6b8` final spec `d95a8bf0…`; Appendix C amended to implicit affine form, no `[D,D]` (C8). Corrections: dual ridge form, SVD convention, Tier-2 estimator, demo M=20, real-domain psi.
- Impl `#6a8cf948`: `HENRI_F7_AFFINE=1` default-OFF; F7 8/8 + F6/F5/arc 35/35; per-file impl hashes recorded.
- **Verdict `FALSIFIED_AT_SCALE` (#177c30b5, commit `4e27730`, split seed 20260902, split sha `015284e0…`):** arms A_affine_full **0.4051** / B_affine_tier1 0.4051 / C_f6_circulant_control 0.436 / D_identity 0.1597; G1 0.25 < 0.99 FAIL; G2 0.4051 < 0.7 FAIL; **G3 margin A−C −0.0309 < 0.25 FAIL**; G4 0.2404 < 0.6 FAIL. Per-env: ft09/lp85 = trivial_single_action; others 0.20–0.31 = no_linear_action_signal.

## 6. F8 — wave-bank decodability probe (`carrier/f8-decodability-probe`)

- Directive 1 `4a5a386b…`; prereg `#364346fea5` spec `acecd53b…`; gates G1 ≥ 0.95 / G2 ≥ 0.60 / G3 ≥ +0.25 / G4 ≥ +0.20; ternary verdict vocabulary; bank pinned: npz `9e3c01b4…`, jsonl `1ca089b2…`.
- Impl `#52a01d6b`: `HENRI_F8_PROBE=1` default-OFF; 10/10 contracts; regression 617 passed / 5 skipped; Ref-3 strip filter committed `3962091b24`.
- Amendment `#470e6b58` (directive 2 `b0d2fe9e…`, commit `85da8bf4…`, seed 20260903): **schema correction — OBSERVED real-domain float16 psi `[1536,65536]`, actions_onehot uint8 `[1536,7]`, jsonl env meta (directive complex-domain assumption falsified)**; loader fail-closed on mismatch.
- **Verdict `F8_INDETERMINATE / CASE_C` (#04df0f9f, CUDA, commit `85da8bf4`, seed 20260903):**

| Probe | CV accuracy |
|---|---|
| LS (min-norm) | 0.3490 |
| Logistic (L2) | 0.3819 |
| MLP | 0.3984 |
| k-NN k=1 | 0.3956 |
| **k-NN k=3 (max)** | **0.4248** |

- Majority baseline 0.235; **nontrivial-env acc_max 0.2449** (trivial envs [3,6] excluded — H=0 single-action loops); TD arm ΔΨ − static = **−0.0078**.
- Gates: G1 train 0.8895 FAIL (cannot overfit); G2 0.4248 FAIL; G3 margin +0.1898 FAIL; G4 TD−static −0.0078 FAIL. Receipt: `/tmp/henri_f8_decodability/f8_gates_receipt.json`.

---

## 7. Cross-cutting synthesis

1. **Identity arm is constant** at 0.1597 across F5–F7 (same bank, same split rule family) — a stable measurement floor, not a moving baseline.
2. **Operator families plateau:** nonlinear head 0.2296 (payload-starved), FPB 0.4352, legacy 0.4414, adaptive-unitary 0.436, affine 0.4051 — persistence across operator families moves the hypothesis from *unbinding operator* to *representation*: action labels are not (non)linearly/affinely decodable from bank waves at these scales.
3. **F8 partial signal is weak and not temporal:** best CV (k-NN 3) 0.4248 is +0.19 over majority but collapses to 0.2449 excluding trivial envs; G4 (ΔΨ derivative ingress) −0.0078 does not support a time-differential mechanism.
4. **Governance posture:** every carrier default-OFF, runner/main untouched, honest negatives sealed as governance records; no capability claim was made from any falsification.

## 8. Sanctioned next steps (each requires a separate directive)

- **Per-env decodability decomposition** — the proposed next falsification: which envs carry any signal at all (F8 report).
- **F8.1 ΔΨ derivative-bank probe** — NOT pre-cleared by G4 (−0.0078 < +0.20); requires its own pre-registration, kill criterion, and fresh corpus.
- **F8.2 supervised egress** — NOT licensed (CASE B unmet: 0.4248 < 0.75).

---

## 9. Provenance

- Ledger path: `C:\Users\chan\AppData\Local\hermes\audit\henri_audit_chain.jsonl`
- Verify receipt: `chain intact: 1043 records, head fb2535a3a8ae0eb8…`
- All event hashes (`#…`) are the `block_hash` values of the corresponding records; all commits are full git SHAs from the payloads.
- This draft is compiled from direct ledger reads only. It becomes a sealed artifact when (a) a `RESEARCH PUBLISH` record with its SHA-256 is appended to the chain, and (b) the file is committed to the repo.
