# Class 4.4 Path B2 — Hard-Negative Discriminative Codec: Bounded Design & Pre-Registration

Doc ID: HENRI-CLASS44-PATHB2-EXECUTION-2026-08-21
Authoritative spec: `/g/My Drive/HENRI_Inbox/Project_HENRI__Isometry_vs._Semantic_Margin_Synthesis___Literature_Integration.md` (SHA `fa145c46…`)
Also: `HENRI_SOTA_VLA_Reverse-Engineered_Roadmap.md` (SHA `6c534e93…`), `HENRI_Model_Size___Zone_C_VRAM_Offloading_Analysis.md` (SHA `916935fb…`), `Zone_C_Memory_Operational_Directive.md` (SHA `c672c28d…`), `Universal_Zone_C_TimescaleDB_Seeding_Module.py` (SHA `ce3c005b…`).

## 1. Academic foundation

Gate A failure of Path B1 (sealed `PATH_B_GATE_A_FALSIFIED`, commit `d90c585` + addendum `6e11bd2`) showed: unit-modulus phasor isometry is NECESSARY but INSUFFICIENT for semantic discrimination. cos(target, goal) = 0.91 but cos(lookalike, goal) = 0.99 ⇒ target ranked #31–32/71. The cause is carrier dominance: the goal wave is dominated by shared surface tokens (AST node types, common keywords) whose similarity overwhelms the discriminative signal of the actual primitive.

Fix direction (per spec): train the ENCODER to deform the local manifold so lookalike mutants are penalized — hard-negative InfoNCE (τ=0.07) over execution-verified grammar mutants from the 71-candidate production pool, with Cholesky retraction after every gradient step (E_Gram < 1e-5).

## 2. Mechanism

`PathB2DiscriminativeCodec(nn.Module)`:
- Tokenization: AST node types + lexical tokens (same as Path B1) + qFHRR-IDF weighting: token weight w_t = idf(t) = log(N / (1 + df_t)); the codec embedding is multiplied by idf before phasor projection.
- Lift: frozen functional block-phasor lift (O(D) storage, no [D,D] matrix) mapping [d_latent] → unit-norm real wave on S^{D-1} (interleaved cos/sin).
- Hard negatives: `_generate_hard_negatives(code, n=70)` — deterministic semantic-preserving mutants (identifier renames, arg-order swaps, binop flips, arity variants) that CHANGE the solution; generated from the production 71-candidate grammar pool by executing each mutant in the sandbox and keeping only execution-verified (mutant passes/fails differently) candidates.
- InfoNCE loss with in-batch + explicit hard negatives; temperature τ = 0.07 (per spec).
- Cholesky retraction: W ← L⁻¹W after each optimizer step on the embedding (Gram error < 1e-5), keeping the codec isometric.

## 3. Data path (zero-pretraining invariant)

- Training corpus: MBPP (data/mbpp.jsonl, 974 items, SHA `ccf64ce…`) — code solutions only, NO held-out HumanEval solutions, NO ARC grids, NO evaluation caches.
- Validation: MBPP split (deterministic seed 7).
- Gate A probe: HumanEval/23 + /35 — measures ranking quality vs the production 71-candidate grammar pool; the ORACLE is used ONLY to measure rank, never in candidate generation or sandbox.
- Gate B (only if A passes): paired HumanEval runner, control OFF vs treatment ON, same dataset/evaluator/hardware.

## 4. Resource limits

- D = 65,536; d_latent = 512; embedding [|V|, 512] (|V| ≤ 4096) ≈ 8.4M params; no dense [D,D] (banned, ~34 GiB).
- Training: 1500 steps × batch 32 on RTX 5090 ≈ minutes; Gate A probe seconds.

## 5. Expected benefit (falsifiable)

- Gate A: oracle rank ≤ 5/71 for BOTH HumanEval/23 and /35 AND margin ≥ +0.25 on RTX 5090.
- Gate B: HumanEval pass rate > 5/50 (> 10%) vs baseline 2/50 (4%).

## 6. Failure modes & kill gates (pre-registered, binding)

| Gate | Criterion | Kill |
|---|---|---|
| Gate A | BOTH ranks ≤ 5/71 AND margin ≥ +0.25 | ANY fail ⇒ revert + seal `PATH_B2_GATE_A_FALSIFIED` + halt (no Gate B) |
| Gate B | HumanEval > 5/50 | ≤ 2/50 ⇒ revert + seal `PATH_B2_GATE_B_FALSIFIED` |
| Infra | CUDA suite fails | classify bucket; retry SAME candidate once isolated |

## 7. Supplied-module audit (Universal_Zone_C_TimescaleDB_Seeding_Module.py, SHA `ce3c005b…`)

Dispositions (per henri-research supplied-artifact audit):
- `encode_text_ast` — DETERMINISTIC SHA-256-seeded AST-walk; same random-ring family as the FALSIFIED qFHRR codec-geometry control (no semantic proximity). `NOT_LOAD_BEARING` for discriminative codec; reuse only its dual-subspace table names.
- Schema DDL (`zone_c_ast_engrams`, `zone_c_action_engrams`) — `ALREADY_SPECIFIED` in the roadmap; adoption requires migration + OID audit. `NOT_EXECUTED` in this phase.
- `encode_action_trajectory` = `_generate_fallback_vector` — RANDOM-RING for actions; `FALSIFIED_GEOMETRY` (same codec-geometry kill). Not used.
- Credentials hardcoded defaults (`postgres/postgres`) — `REQUIRES_APPROVAL`; never used in this phase.
- Zero-pretraining: seeding module would ingest `domain="arc"` items into zone_c_action_engrams — PROHIBITED for evaluation content. Corpus seeding deferred to a gated later phase with a new bounded design.

## 8. Execution order (this phase, approved)

1. Contract tests (RED) → 2. codec + trainer + probe + runner wiring (default-OFF `--path-b2-codec`) → 3. GREEN contracts + py_compile → 4. commit + push → 5. remote exact-SHA worktree + overlay + full CUDA suite → 6. train at D=65536 → 7. Gate A → (pass ⇒ Gate B paired run) → 8. verdict + revert/seal per gates.

No main changes; no live Zone C writes; no evaluation-content ingestion.
