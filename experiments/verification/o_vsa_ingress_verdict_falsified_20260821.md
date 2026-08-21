# O-VSA Stage 1 Verdict — FALSIFIED (Class 4.6)

**Doc ID:** HENRI-CLASS46-O-VSA-INGRESS-FALSIFIED-2026-08-21
**Packet:** `experiments/verification/o_vsa_stage1_gate_20260821.md`
**Spec:** `HENRI_Ontological_Phase_Manifold_Remedy_Specification.md` SHA
  `f9cef399082e90419683b7fd2fdf716491aa7d8f181506b57aa14509b20738ee` (9,743 B, frozen copy in
  `experiments/verification/o_vsa_remedy_spec/`).
**Candidate SHA:** `678caf720d7d1a975fb908fc1f708ecb13721a29` (pushed, remote exact-SHA worktree,
  zero status lines, GPU idle pre-launch).

## Decisive measurement (OBSERVED, RTX 5090, D=65,536, CUDA, dataset SHA b796127e635a67f9)

Paired arms at the same SHA / pool / hardware. Receipt:
`experiments/verification/o_vsa_ingress_evidence/o_vsa_gate_a_d65536_678caf7.json`
(SHA-256 `ea5f8dfa33a0b88af0cba34298ccade1667c15d623e6a178df11e97571222d06`),
log `..._678caf7.log` (69 lines).

| Target | Arm | oracle rank (≤5) | true cos | best cross cos | margin (≥0.25) |
|---|---|---|---|---|---|
| HumanEval/23 | random_ring (control) | 18 | 0.0020 | 0.0106 | −0.0086 |
| HumanEval/23 | **o_vsa** | **70** | 0.067 | 0.557 | **−0.490** |
| HumanEval/35 | random_ring (control) | 8 | 0.0067 | 0.0108 | −0.0040 |
| HumanEval/35 | **o_vsa** | **66** | 0.125 | 0.606 | **−0.481** |

## Verdict

- T1 semantic engagement: PASS at D=65,536 (related cos 0.7003 len/count, 0.6609 max/maximum
  ≥ 0.40; unrelated 0.1253/0.0864 < 0.15).
- T2 production ranking: FAIL on both targets (rank 70/66 > 5; margin −0.49/−0.48 < 0.25).
  O-VSA ranks WORSE than the random-ring control (18/8).
- Kill rule fired per packet: **O_VSA_INGRESS_FALSIFIED** — no Stage 2 (Lan_K), no Stage 3
  (Hopfield settling), no external HumanEval gate, halt.

## Root cause (DERIVED from receipt + direct diagnostic)

Mean-phasor bundling with equal token weights makes the O-VSA encoder sensitive to SHARED
lexical tokens between goal prompt and candidate bodies (function signature, arg names,
docstring words). Those shared tokens act as a carrier: every candidate scores 0.55–0.61 vs
the goal while the true body's distinctive `len`/`max` is drowned (true cos only 0.07/0.12).
Body-only T1 pairs (no signature carrier) pass; real prompt-vs-candidate pairs fail. This is
carrier dominance re-expressed INSIDE a structured encoder; weighting/comb design must be
addressed before any re-attempt.

## Dispositions

- `--o-vsa-harmonic-ingress` stays **default-OFF**; module + probe retained as sealed evidence
  (flag never changes production behavior).
- No Stage 2/3 without a NEW pre-registered packet.
- Remote worktree `/root/henri-o-vsa-678caf7` removed after evidence retrieval.
- Path B2 / HOPS-VSA seals remain immutable.
