# Class 4 Path A — In-Context Demo-Pair Operator Design Boundary

Status: IMPLEMENTATION READY (authorized 2026-08-20)
Base: `accuracy/fidelity-remediation@107e612`
Spec: `HENRI_Path_A_Inference_Mechanics_Assessment.md` (drive inbox, 2026-08-20)
SHA-256 (spec): `G:\My Drive\HENRI_Inbox\HENRI_Path_A_Inference_Mechanics_Assessment.md`
Implementation: `HENRI V2/henri_task_operator.py` (factorized R-EDMD)

## Authorization

User directive (2026-08-20, Photon): attempt Path A first; Path B only if
Path A is insufficient. Path A keeps the zero-pretraining invariant.

## Mechanism (5 steps, per spec)

1. Ingress: authorized in-context `(X_i, Y_i)` pairs → `S^{D-1}` waves via
   `ASTDiscriminativeEncoder` (structural AST phase encoding).
2. Cross-covariance: least-squares task operator `W_task` in closed form
   `Y_c X_c^+` (thin SVD; no `O(D^3)`, no dense `[D,D]`).
3. Unitary projection: Newton-Schulz iterations on the `[D, r]` left
   factor (r×r Gram core, ≤5 iters, quadratic convergence).
4. Inference: `Psi_goal = W_task x_query` (factorized `A (U^T x)`).
5. Egress: reorder the EXISTING grammar candidate set by phase alignment
   `<Psi_c, Psi_goal>`; sandbox decides PASS/FAIL.

## Authorized demo source (HumanEval)

The prompt's own docstring `>>>` examples — part of the task specification
the model receives. NEVER the `test` field; NEVER the reference answer.
Compiled 100% online at test time. Zero-pretraining invariant preserved.

## Factorization contract (architecture invariant)

- No `[D, D]` tensor (≈34 GiB at D=65,536). Storage: `[D, r]` + `[r]` only.
- `_assert_factor_shape` rejects dense square intermediates at every site.
- Apply path allocates `[D, r]` / `[r]` / `[D]` intermediates at most.

## Pre-registered gates (before remote CUDA run)

Run: HumanEval 50-item slice, `--path-a-demo`, exact candidate SHA on Vast
5090, official evaluator, zero infrastructure errors.

| Gate | Accept | Kill |
|---|---|---|
| G1 causal engagement | flag ON changes candidate order vs OFF on ≥1 item | order byte-identical on all items |
| G2 demo recovery | ≥1 item has docstring `>>>` pairs extracted | 0 items have extractable demos |
| G3 operator validity | rank ≥2, orth_error < 1e-3, no dense alloc | rank < 2 or dense `[D,D]` asserted |
| G4 external outcome | ≥1 NEW pass attributable to Path A reorder (passing candidate was not first in grammar order) | 0 new passes after bounded run |
| G5 ranking fidelity | on items where the true solution is in grammar: true_rank ≤ 2 AND margin ≥ +0.05 | true_rank > 2 or margin < +0.05 |
| G6 integrity | zero infra errors, zero leakage, zero score promotion without profile | any infra error or leakage → `BLOCKED_INFRASTRUCTURE` |

Interpretation rules:
- G4/G5 pass → next bounded tracer (e.g. 164-item full HumanEval, MBPP).
- G4 fail but G5 pass → rank signal works, egress insufficient: extend
  grammar under the SAME operator (bounded, pre-registered).
- G5 fail → representation operator FALSIFIED: seal `PATH_A_RANK_FALSIFIED`,
  revert wiring, evaluate Path B authorization.
- Any infrastructure failure blocks the whole verdict (fail closed).

## Telemetry (scorecard fields)

`path_a_demo_enabled`, `path_a_items_ranked`, `path_a_items_no_demo`,
`path_a_items_underdetermined`, `path_a_new_passes`,
`execution_profile`, `score_promotable`.

## Boundary: sealed levers stay sealed

Path A is a NEW semantic representation path (authorized). The sealed
ranking levers (reward/decoder/spec/trained/ast-idf) remain CLOSED and fail
closed under `HENRI_ACCURACY_FIRST_CLASS4`. Path A does not reopen them.
