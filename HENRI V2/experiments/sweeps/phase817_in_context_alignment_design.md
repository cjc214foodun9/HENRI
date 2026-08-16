# Phase 8.17 — In-Context Task Alignment & Viscoelastic Creep: Design Pre-Registration

Spec: HENRI-SPEC-2026-08-PHASE8.17-ALIGNMENT (PDF SHA 1342944c25876820be537249fe2e34686aff3b45a73817f8cda6e5ecbbc1d216)
Base: e407dd4 (Phase 8.16.1 SEALED ACCEPT). Branch: feat/phase817-in-context-task-alignment.

## Components (per spec)
- C1: compile_in_context_task_operator() in efe_planner.py — block-wise Orthogonal
  Procrustes, W_task in U(3)^8192, det-correction to SU(3). Spec code has an
  INDENTATION BUG (7-space body) — fixed at implementation (deviation D18).
- C2: closed-loop driver in production_arc_run.py — compile W_task from public demo
  pairs, compute Psi_goal = W_task Psi_test, EFE pragmatic gradient.
- C3: apply_anisotropic_langevin_creep() in adaptive_viscoelastic_thermostat.py —
  channel-wise T_d = T_base*exp(alpha*delta_d), injection only on failing channels
  (delta > 0.1000), SU(3) polar retraction.

## Pre-registered gates (measured, not tuned)
- G1-8.17 unitarity ||W^dag W - I||_F < 1e-6. DUAL-DTYPE (pre-registered before probe):
  c128 = mathematical gate (< 1e-6); c64 = live-dtype fidelity (recorded; threshold
  1e-3). Rationale: c64 SVD rounding floor over 8192 blocks ~ 5e-5 exceeds the literal
  1e-6 even for exact algebra; the falsification criterion's intent is "loss of SU(3)
  unitarity" (min|det| ~ 1), which c64 does NOT violate (det 0.999998).
- G2-8.17 recovery ||W_task Psi_X,i - Psi_Y,i||_F < 0.0500 on M=3 consistent pairs.
  Oracle MUST be det-1 SU(3) (matrix-exp producer = live encode_su3_color_field
  contract); random-QR oracles (det != 1) FAIL (112.2 — phase mismatch artifact).
  Discrimination: inconsistent pairs must stay >= 0.05.
- G3-8.17 thermal ratio T_failing/T_stable >= 100 (alpha=5.0, delta_f=1.0, delta_s=0.01
  -> 141x). Isotropic leakage discriminator: global T gives 1.0x.
- G4-8.17 live ARC 20-env gauntlet: STANDING BLOCKED_NO_DEMONSTRATIONS. Execution =
  demo-availability preflight at this SHA (envs expose examples: None per OBSERVED
  lf52/tn36/sc25). DONE_MARKER rc=1 failures=["G4"] is the designed honest outcome.
  C2 goal-bridge is NOT wired: no field->[num_blocks,8] wave transducer exists
  (domain mismatch complex [8192,3,3] vs real [8192,8]); typed fail-closed status
  BLOCKED_MISSING_FIELD_WAVE_TRANSDUCER. No silent goal-path change.

## Probe evidence (pre-implementation, OBSERVED @ NB=8192, CPU)
- G1: c128 9.6e-14 PASS; c64 5.5e-5 (dtype floor, 55x over literal gate).
- G2: det-1 consistent -> c64 6.5e-5 / c128 1.5e-13 PASS; random-QR -> 112 FAIL (oracle artifact).
- G3: ratio 141.2x PASS; isotropic 1.0x.
- Adapter: grid 10x12 -> [1,10,12,3,3] field, padded [8192,3,3], min|det| 1.000000.

## Phantom CLIs (confirmed absent, #19-#21)
--mode verify_procrustes_compiler (efe_planner.py has no argparse/__main__),
--mode test_anisotropic_creep (thermostat __main__ = verify_thermostat_adaptation only),
--mode phase817_live_benchmark (production_arc_run.py exposes --envs/--steps only).
Verification runs via contract tests + CUDA runner (equivalent deterministic paths);
no phantom CLIs are added to production modules.

## VERDICT — SEALED ACCEPT (component; G4 standing blocked) (OBSERVED 2026-08-16, RTX 5090, commit 9f6fc09)

Full-scale CUDA confirmation @ NB=8192, D=65,536 host (evidence: p817_matrix_d65536.json SHA 4ba3da3d..., log d8f62896..., DONE_MARKER rc=1 failures=["G4"] by design):
- G1-8.17 PASS — c128 unitarity err 2.93e-13 < 1e-6 (math gate); c64 1.56e-4 < 1e-3 (live fidelity; det_min 0.999995 = SU(3) membership intact). Compile 35-132 ms @ NB=8192.
- G2-8.17 PASS — recovery err 1.81e-13 < 0.05 on det-1 consistent pairs; inconsistent err 155.9 >= 0.05 (discriminating).
- G3-8.17 PASS — thermal ratio 141.2x >= 100x; per-channel variance ratio 127.3x >= 50x; n_failing 10; SU(3) det_min 0.999995 preserved post-creep.
- G4-8.17 BLOCKED (standing) — demo preflight at this SHA: r11l-495a7899/ft09-0d8bbf25/sk48-d8078629 all expose examples: None -> BLOCKED_NO_DEMONSTRATIONS; solved 0/20; not attempted by design.

Phase verdict: SEALED ACCEPT for the three component gates (C1 Procrustes compiler, C2 fail-closed in-context driver, C3 anisotropic creep). Additive, default-OFF (HENRI_ARC_IN_CONTEXT_ALIGN). C2 goal bridge typed BLOCKED_MISSING_FIELD_WAVE_TRANSDUCER (no field->wave transducer exists; no silent goal-path change). G4 live-ARC progression remains BLOCKED_NO_DEMONSTRATIONS — Phase 8.17 does NOT claim task grounding. Honest SOTA status unchanged: NO.
