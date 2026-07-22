---
id: "low_rank_coupling_constraints"
created_at: "2026-07-18T09:30:00Z"
updated_at: "2026-07-18T09:30:00Z"
module: "Low-Rank Coupling"
tags:
  - henri-v2/theory
status: "verified"
---

# Low-Rank Transition Matrix Constraints

The EDMD transition operator is stored in factored low-rank form (rank r=64).
Near-unit eigenvalues mark invariant constraint modes. The off-manifold
residual must enter scoring as a per-candidate cost term, never as an
additive channel — an additive min-over-channels objective can only attract,
turning the residual into a false attractor (run 12 falsification).
