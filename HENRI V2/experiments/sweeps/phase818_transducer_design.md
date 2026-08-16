# Phase 8.18 — Field-to-Wave Isomorphic Transducer: Design Pre-Registration

Spec: HENRI-SPEC-2026-08-PHASE8.18-TRANSDUCER (PDF SHA 158c02c7be323165f497969cf3387137a3f5f3696dab669207616b55be50d449, 307 lines)
Base: b71fe4c (8.17 seal). Branch: feat/phase818-su3-field-wave-transducer.

## Mechanism
C1 SU3FieldWaveTransducer (universal_data_transducer.py): matrix log of SU(3) -> su(3) via
eigendecomposition (D19: torch.linalg.matrix_log absent on torch 2.12.0+cu130, VERIFIED remote),
Gell-Mann projection theta_a = 0.5 Re Tr(lambda_a A) (D20: spec einsum 'abc,bncb->bna' has b-index
collision; correct 'aij,bnji->bna'), flat [8192,8] -> exp(i theta) unit-modulus complex wave [65536].
Inverse: angle -> theta -> sum theta_a lambda_a (D21: spec 'bna,abc->bnabc' never contracts the
generator index; correct 'bna,aij->bnij') -> matrix_exp. Gell-Mann basis taken from live
chromodynamic_grounding.GELL_MANN_BASIS (single source).
C2 production_arc_run.py: replace BLOCKED_MISSING_FIELD_WAVE_TRANSDUCER fail-closed status with the
transducer bridge: field_to_wave(W_task @ U_test) -> angle() -> real [num_blocks,8] goal wave ->
orch.planner.lambda_goal already set when LAMBDA_GOAL>0. Default-OFF via HENRI_ARC_IN_CONTEXT_ALIGN.
Goal bridge status W_TASK_GOAL_BRIDGED (telemetry only). No score-eligibility change.
C3 qfhrr_kernels.py: triton.jit su3_log kernel. Closed-form matrix log for 3x3 via eigendecomposition
is NOT tractable in Triton (no complex eig); the spec's "characteristic polynomial roots" closed form
for a 3x3 matrix log exists for diagonalizable matrices with distinct eigenvalues:
log A = sum_k f(lambda_k) prod_{j!=k} (A - lambda_j I)/(lambda_k - lambda_j). Characteristic polynomial
of a 3x3: p(x)=x^3 - tr x^2 + (1/2)(tr^2 - tr A^2) x - det. Per-block tr/det/trace-square are cheap
(load 9 c64, 3x3 products in SRAM). Root-finding in Triton is not supported -> implement the
eigenvalue computation as closed-form cubic roots using the trigonometric (Cardano) method in tl
arithmetic (real arithmetic on the Hermitian cas... `-i log U` is Hermitian: real eigenvalues,
closed-form cubic via cos/acos — Triton supports tl.cos/tl.acos). Then Lagrange interpolation
per eigenvalue. Blocks: grid (8192,), each program handles one 3x3. Latency gate <= 50 us (table)
per spec table (D22: SS2.3 prose says 45us, table G3 says 50us — table wins).

## Gates (pre-registered, spec-mandated)
- G1-8.18: mean per-block round-trip ||U - Phi^-1(Phi(U))||_F < 1e-5. Probe (CPU @NB=8192): c64 8.19e-7 PASS, c128 1.5e-15 PASS.
- G2-8.18: ||Phi(UA UB) - Phi(UB UA)||_2 > 0.5. Probe: 227.82 PASS; commuting control 2.1e-3 (discriminates; not vacuous).
- G3-8.18: Triton su3_log kernel sustained latency <= 50 us (CUDA-event interval, 20 back-to-back launches).
- G4-8.18: live ARC >0 solved. Pre-registered BLOCKED: public demo pairs are None across arc_agi
  environments (OBSERVED r11l/ft09/sk48; 8.17 preflight) -> BLOCKED_NO_DEMONSTRATIONS standing.
  Falsification = persistent 0/20. No pseudo-demos.

## Failures / deviations
- D19: matrix_log -> eigendecomposition log (both local + remote torch lack the API; VERIFIED OBSERVED).
- D20: field einsum 'abc,bncb->bna' -> 'aij,bnji->bna' (b-index collision would RuntimeError).
- D21: wave einsum 'bna,abc->bnabc' -> 'bna,aij->bnij' (spec never contracts -> would produce [B,N,8,3,3]).
- D22: G3 threshold conflict 45us (prose) vs 50us (table) — table wins (20 kHz budget = 50us).
- Phantom CLIs #22 gpu_verification_suite.py (3rd citation), #23 --mode verify_su3_transducer, #24 --mode phase818_live_benchmark.

## Kill criteria
- G1 or G2 fail at CUDA full scale -> KILL (algebra not isomorphic as specified).
- G3 sustained > 50us -> KILL (Triton kernel not meeting budget; fall back to torch path for C2).
- G4 BLOCKED is expected; not a kill. ACCEPT = C1/C2/C3 component gates.
