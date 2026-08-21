# Path A Verdict: FALSIFIED_NO_EXTERNAL_GAIN (with regression) — 2026-08-21

## Status
SEALED. Path A (in-context demo-pair operator, `dab6e10`) is FALSIFIED as a
HumanEval accuracy improvement and is REVERTED. The commit remains in branch
history as governance evidence. Default path is unaffected (flag default-OFF
was already in place; the revert removes the operator and wiring entirely).

## Paired gate (pre-registered in class4_path_a_operator_design.md G1–G6)

Same candidate `dab6e10`, same dataset (HumanEval.jsonl.gz SHA-256
`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`), same
evaluator, same split (50 items), same attempts budget (12), same hardware
(Vast instance 47411800, RTX 5090), sequential arms, zero infra errors.

| Arm | solved | expressible | infra_errors | path_a_items_ranked | path_a_new_passes | scorecard SHA-256 |
|---|---|---|---|---|---|---|
| Control (Path A OFF) | **2/50** | 50 | 0 | 0 | 0 | `ef04122f…` |
| Treatment (--path-a-demo) | **0/50** | 50 | 0 | 31 | 0 | `a9182eac…` |

Raw logs: control `af6e1bf08b2f…`, treatment `ee0dc55c25…` (preserved).

## Item-level evidence

- Control passes: `HumanEval/23`, `HumanEval/35`.
- Both regressed in treatment: `path_a_engaged=True`, `rank=2`,
  `demo_mse=0.0`, `a_orth_error=1.476` / `1.194`, singular values
  `[1.414, 0.0]` / `[1.100, 0.888]`.

## Math verification conclusions (per /henri-research request)

1. **Least-squares operator is NOT unitary** (unitary-claim trap confirmed):
   `a_orth_error` 1.19–1.48 on live items proves the `A` output factor is
   unconstrained; only `U` (input factor) is orthogonalized. The operator is a
   partial isometry on the demo span at best.
2. **Exact-demo memorization, not compositionality**: `demo_mse=0.0` with
   rank-2 and 2 demos is a perfect fit on the demo span; held-out behaviour is
   what matters and it failed.
3. **Coverage was adequate but shallow**: 31/50 items ranked, 15
   underdetermined (<2 pairs), 4 no-demo. Compile failures 0.
4. **Engagement ≠ outcome**: candidate order changed (rank 2 for the two
   previously-passing items) yet the reorder pushed both beyond the attempt
   budget → 2 → 0. Ranking levers remain closed on HumanEval (consistent with
   the 2026-08-20 Gate A' transfer kill: bottleneck is grammar expressiveness,
   not candidate order).

## Governance

- Verdict: `FALSIFIED_NO_EXTERNAL_GAIN` (regression observed).
- Action: `git revert dab6e10` applied; evidence committed; branch pushed.
- Next candidates (Path B or representation work) require a new pre-registered
  design packet before implementation.
