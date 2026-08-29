# Gap 5 Carrier Preregistration — External-Outcome → Zone C Crystallization (DRAFT, REQUIRES APPROVAL)

Spec ID: SPEC-2026-08-29-GAP5-CRYSTALLIZE
Status: DRAFT — bounded design for approval; not yet implemented.

## Mechanism (hypothesis, falsifiable)

Gap 5 of the VLA convergence doc (`81cd8a33…`): "continual learning — Δν →
Zone C crystallization missing" (disposition `BOUNDED_IMPLEMENTABLE`).

When the live runner observes a strictly positive external outcome delta
`Δν_t = Score_{t+1} − Score_t > 0` on a task episode (verified progress only —
the existing Beta-Bernoulli task store already gates `verified progress`), the
post-transition wave state `Ψ_{t+1}` (canonical `[8192,8]` Cl(3,0) block wave,
float32) is crystallized into Zone C as a **frozen reference engram** with full
attribution (`run_id`, `arm_id`, `commit_sha`, `domain_family` per CLASS49
Gate-1 semantics) and retrieval isolation (`domain_family` filter per Gate 4).

This is NOT learning new weights. It is a test-time memory write gated by
external outcome — the boundary-axiom pattern already established
(`zone_c_axiom_seeder.py`) extended to episodic success.

## Gates (default-OFF `HENRI_ZONE_C_CRYSTALLIZE=1`)

- G1 (engagement): ≥1 crystallization write on a corpus with ≥1 verified
  progress event; zero writes on a matched Δν≤0 control corpus.
- G2 (attribution): every write persists run_id/arm_id/commit_sha/domain_family
  to SQL columns (no `**kwargs` swallow); recall enforces the same fields.
- G3 (retrieval): crystallized state retrievable by `domain_family` filter,
  P@1 ≥ 0.99 against the written wave (self-consistency, not capability).
- G4 (dense-ban): payload is `[8192,8]` wave bytes (num_blocks×8×4), no dense
  `[65536,65536]` allocation anywhere in the write path.
- G5 (eligibility): crystallization alone NEVER grants score eligibility
  (fail-closed; `score_eligible` unchanged by this carrier).

## Kills

- K1: zero writes despite verified progress → `FALSIFIED_NO_ENGAGEMENT`.
- K2: writes on the Δν≤0 control → `FALSIFIED_CONTROL_CONTAMINATION`.
- K3: attribution fields missing in any written row → `FALSIFIED_ATTRIBUTION`.
- K4: no retrieval isolation → `FALSIFIED_ISOLATION`.

## Boundary

- Frozen encoder/learner untouched; this carrier only adds a gated write path.
- No pre-training, no task-data pre-ingestion (zero-pretraining invariant intact:
  crystallization happens only from LIVE episode outcomes).
- Schema change to Zone C requires the canonical migration path
  (`migrations/zone_c_schema.sql`), never ad-hoc DDL.
- Promotion is a separate approval gate after CUDA verification.

## Cheapest kill experiment

Control corpus with forced zero external gain (frozen learning, same envs/seeds)
must produce zero writes. If any write occurs, the gate is broken — no
capability claim possible.
