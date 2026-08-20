# Phase 8.39 — Gate A′ Pre-Registration (IDF-Only Representation)

Specification Identifier: **HENRI-SPEC-GATE-A-PRIME-IDF-2026**
Mirror of drive-inbox doc `Class_3.0_Findings___Gate_A__Pre-Registration.md` (Aletheia).
Branch: `phase839/humaneval-wave-ast`. Sealed at `2bfb71b` BEFORE any Gate A′ execution.

## 1. Falsifiable hypothesis
IDF phasor magnitude weighting at AST ingress suppresses ubiquitous structural syntax
(Module/FunctionDef/Return/Name/Load) while preserving essential operator relationships,
raising I(Y; Ψ_AST) so that correct solution bodies rank ≤ 5/71 in the production
71-candidate grammar pool, WITHOUT egress orthogonalization (E[cos] ≤ 0.10 is NOT required).

## 2. Controlled mechanism
- Lever 3.2-A (ENABLED): ingress IDF phasor accumulator. w(op) = log(1 + N/f(op)) scales
  each node-type phasor before depth/child-position fractional binding and unit-modulus
  quantization to Z_256^D. Unseen node types receive max-IDF (never 0).
- Lever 3.1-A (DISABLED): carrier subtraction is REMOVED from the execution path.
  No Gram-Schmidt projection of the MBPP carrier.

## 3. Proxy substrate (pre-registered)
- CPU probe, d = 2,048 (cheapest-kill proxy; Class 2.0/3.0 precedent).
- Candidate pool: production grammar pool via `WaveASTDecoder._instantiate` (71 candidates
  for both items; real grammar tables).
- Attractor bank: MBPP codebook N=100, canonical `data/mbpp.jsonl`
  (SHA-256 prefix `ccf64cea…`), same configuration encoding.
- Ranking: mean raw phase-cosine vs codebook bank (Class 2.0 Lever 2.2 precedent).
- Oracle items: HumanEval/23 `return len(string)`, HumanEval/35 `return max(l)`.
  Correct-body match by exact body text (docstring-only prompt; no answer-key leakage).

## 4. Gate A′ evaluation rules
| Check | Rule | Action |
|---|---|---|
| M2 /23 | rank ≤ 5/71 | required for PASS |
| M2 /35 | rank ≤ 5/71 | required for PASS |
| M1 E[cos] | recorded only, NOT a kill gate | telemetry only |
| Kill | either rank > 5/71 | FALSIFIED; Gate B skipped; verdict sealed |

Exit: 0 = PASS (proceed to Gate B), 1 = FALSIFIED (kill).

## 5. Gate B launch conditions (pre-registered, RTX 5090)
- Triggered iff Proxy Gate A′ passes.
- 50-item HumanEval sweep on host GPU (D = 65,536) with `--ast-idf-only`.
- Target: > 2/50 (> 4.0%) authentic passes (sandbox-executed).
- Falsification: ≤ 2/50 → lever class FALSIFIED at benchmark level; sealed.

## 6. Telemetry schema (receipt JSON)
spec_id, arm=idf-only, flag=--ast-idf-only, d_model, codebook_n, mbpp_sha,
ranks{/23,/35}, e_cos, deterministic, verdict, commit, command, timestamp, gate_b_condition.

## 7. Governance
- Seal event for this pre-registration (event id recorded below after emission).
- Gate A′ verdict event after execution.
- Default-OFF flag semantics preserved: harness without `--ast-idf-only` runs the
  Class 2.0 control arm and applies NO gate verdict.
