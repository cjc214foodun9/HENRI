# Epistemic Forensic Audit — Functional Goal-State Gap Analysis

**Date:** 2026-09-02 · **Auditor:** HENRI arbiter (aggregator)
**Ledger event:** `EPISTEMIC_FORENSIC_AUDIT` `#36d536c1` @1,216

## Ground-state (all OBSERVED)

| Surface | State |
|---|---|
| M1 tip `3d519d2` | P1 goal-steering policy + M1 Δν measurement fix; P1 tests 6/6 within 37/37 CUDA |
| `origin/main` `f46cb17` | ancestor of carrier tip; 0 unique commits (carrier lineage descends from main) |
| Checkpoint overlay | egress decoder head `75572389…` (799 MB) on Vast persistent workspace; **absent in clean worktree** (external overlay convention) |
| Active service | **NONE** — no HENRI process running (`BLOCKED` deployment; jupyter/supervisord only) |
| Ledger | 1,216 verified records, chain intact |
| Untracked artifacts | `henri_audit_chain.json` ×2 (root + `HENRI V2/`; stale repo-local ledger projections, Aug 27/31 — NOT the live JSONL chain; classify as legacy, do not commit) |

## The gap (epistemic bisection, 29 carriers)

1. **Engagement SOLVED** (P1): ΔV(a) potential-drop goal term discriminates actions (mean drops +0.048…−0.027).
2. **Action→outcome coupling NEVER MEASURED**: every G4→G7→P1 run reported `mean_delta_nu_wp: 0.0` through a structurally broken meter (stale `psi64`). M1 repaired the line at `3d519d2`.
3. **Open question (decisive):** along P1's actual trajectory, does ν move toward waypoints at all?

## Goal-state reverse-review (functional HENRI VLA)

- **Substrate family in hand (score high on: representation compat, action conditioning, zero-pretraining, live caller):** frozen boundary-axiom baseplate + UWE `[8192,8]` ingress + R-EDMD dynamics + EFE/ΔV(a) policy + trajectory-bank goals + CEGIS/sandbox egress. This IS the live lineage G1→P1.
- **Capability chain missing (measured):** live action → frame delta → Δν>0 → waypoint advance → `levels_completed` > 0. No carrier has ever observed a positive link past frame delta.
- **Dispositions:** Δν measurement = REPAIRED (M1); goal engagement = CONFIRMED-ENGAGED (P1); outcome coupling = UNMEASURED → P2-0; internet pretraining / new backbones = BLOCKED (zero-pretraining invariant); egress decoder overlay = DIAGNOSTIC/legacy boundary, not in the ARC loop.
- **Smallest decisive carrier: P2-0** — rerun the sealed P1 configuration (same 12 envs × 150 steps, seed 20260930, bank pinned) at M1 tip. Same seed + same policy = same trajectory; ONLY the measurement changed → any Δν≠0 is attributable to the M1 repair. This converts P1's falsification into a measurement-resolved verdict.
