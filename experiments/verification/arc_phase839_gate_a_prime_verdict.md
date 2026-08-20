# Phase 8.39 — Gate A′ Verdict: IDF-Only Representation

Spec: `HENRI-SPEC-GATE-A-PRIME-IDF-2026` (sealed `arc_phase839_gate_a_prime_preregistration.md`)
Status: **PASS** — Gate B launch condition met.
Commit (spec + harness): `5851150`. Runner wiring: pending commit (see below).

## OBSERVED (CPU proxy, d=2048, isolated Python 3.14, torch CPU)
- MBPP corpus: 974 records, 974 usable, SHA-256 prefix `ccf64ceae9c5403b`.
- IDF histogram: 76 node types across 974 ASTs; top: Name:12192, Load:11260,
  Constant:3653, Store:3131, BinOp:2086, Call:2013.
- Candidate pools: HumanEval/23 (`strlen(string)`) pool=71; HumanEval/35
  (`max_element(l)`) pool=71 (production grammar via `WaveASTDecoder._instantiate`).
- Codebook: N=100, 100/100 encoded (IDF-weighted, carrier subtraction DISABLED).

## Gate A′ metrics
| Metric | Rule | OBSERVED | Verdict |
|---|---|---|---|
| M2 HumanEval/23 rank | ≤ 5/71 | **3** | PASS |
| M2 HumanEval/35 rank | ≤ 5/71 | **5** | PASS |
| M1 E[cos] | recorded only | 0.3484 | telemetry (NOT gated) |
| determinism | must hold | True | PASS |

`GATE_A_PRIME=PASS`, harness rc=0. Receipt:
`experiments/verification/arc_phase839_gate_a_prime_receipt.json`.

## Gate B (pre-registered, triggered)
- 50-item HumanEval sweep on RTX 5090 (D=65,536), `--ast-idf-only`.
- Target: > 2/50 (> 4.0%) authentic passes; falsification: ≤ 2/50.
- Launched after remote deploy + dimension-scaling confirmation.

## Governance
- Pre-registration seal event: `2012fc9c-cd38-4384-b5fd-6abd653698d2` /
  AUDIT_HASH `2aefe1c3a05c09120a4280598f157dd47be81d38fa3fa899f6ace5eebb495453`.
- This verdict event: emitted alongside this file (id in commit message).
