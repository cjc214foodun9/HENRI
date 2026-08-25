# Corpus Consult #30 — Task Embeddings Conditioning Structural AST Generation (INFERRED)

**Date:** 2026-08-24 · Notebook: HENRI philosophy (`ca4bb787…`) · Conversation `3179135d-9cca-42a0-a626-3b192ea55cc2`
**Question (one focused):** how should frozen semantic embeddings condition structural AST generation so they change STRUCTURAL SUPPORT and first verified rank — not rerank decoratively?

## Answer (INFERRED — corpus synthesis, NOT OBSERVED telemetry)
- **Reranking candidates after generation is a DECORATIVE intervention** — it alters presentation order but leaves the structural support (the generative grammar's containment set) unchanged (source 4c665d9f…).
- Task conditioning must act as a **universal property / compiled operator** that restricts the admissible search space BEFORE generation unrolls: `W_task = Σ Ψ_Yi ⊛ Ψ_Xi†` compiled from demonstration pairs, applied as `Ψ_goal = W_task ⊛ Ψ_Xtest` (sources ec42ce06…, 8c1c2845…, dbac8042…).
- FunctorFlow framing: task = functor `F_task: C_input → D_output`; the compiled operator acts as Right Kan Extension (limit-based, constraint-aware repair/completion) (sources d8a89525…, 56cfbe73…).
- Goal wave `Ψ_goal` acts as a physical attractor giving a true directional gradient to the EFE planner's pragmatic term `D_goal(Ψ_hat) = 1 − |(1/D) Ψ_hat† Ψ_goal|` (sources 8c1c2845…, 903a914b…).
- Structural filtering via Sagnac veto (Δ > 0.35 → Q → −∞) prunes invalid AST branches BEFORE sandbox; exact-replay CEGIS verifier admits only on replay of recorded transitions (sources fbc0529b…, 56cfbe73…).
- Failure modes documented: ungrounded identity goal → D_goal hovers at 1.0 (static offset, no steering) (ec42ce06…); PEARL repair loop for off-manifold residuals (833bbe68…, b7bd4ca6…).

## HENRI disposition (live telemetry OVERRIDES corpus values)
- Corpus predicts reranking-only is decorative; live Egress-1 result (COST_EFFECTIVE): conditioning changed FIRST VERIFIED RANK (mean 3.23→3.00, rank-0 80→120) and verifier calls (−5.45%, CI lb>0) — so a reorder of the SAME pool (grammar cardinality-bound) is NOT decorative when admission = first verifier pass: it changes which candidate the verifier admits first.
- Corpus's W_task/Ψ_goal directional-steering machinery (MCTS, Sagnac veto, EFE pragmatic term) was NOT used in Egress-1 (reorder-only by sealed grammar cardinality) — that is a separate carrier, not yet tested.
- Structural support (pool CONTENT) could NOT change in Egress-1: skeleton generator saturates at 4–9 unique candidates; gen_task bodies fixed per family (OBSERVED 2026-08-24). Corpus's "restrict the admissible search space before generation" remains the design target for Egress-2 (richer grammar → expansion becomes feasible).
- Sources: 833225d5, 4c665d9f, bac90d63, 90eb8d20, d8a89525, 64993244, ec42ce06, 97b163c4, 8c1c2845, 56cfbe73, 0b7d4b6a, 21e2cb54, dbac8042, 7413e9aa, 903a914b, f5a32da3, b7bd4ca6, 833bbe68, cdadb1cb, b1311ac2, fbc0529b, 8af1a9eb, 09d5305f.
