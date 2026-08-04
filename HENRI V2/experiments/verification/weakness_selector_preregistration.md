# Extension-Mass Selector: Pre-registration

Status: approved, default-off, experimental, not benchmark evidence.

## Hypothesis

For a finite candidate continuation set `E(c)`, the exact extension mass
`W(c) = |E(c)|` can break a true baseline EFE tie without changing a
non-tied decision.

This is a finite extension experiment. It is not a proof that the HENRI wave
core implements Bennett weakness, and it does not infer extension mass from
text length, entropy, similarity, or a later environment outcome.

## Scope

- New deterministic module: `HENRI V2/weakness_selector.py`.
- Optional planner input: `weakness_extensions`, aligned with candidates.
- Default: disabled.
- No changes to wave mechanics, Sagnac thresholds, Zone C, Hopfield storage,
  decoder checkpoints, benchmark evaluators, or external outcome updates.
- No implementation may construct extension sets from future data.

## Controls

1. Existing EFE ranking.
2. Complexity or output-length metadata, used only as a negative control.
3. Extension mass used only within `weakness_tie_tolerance` of the baseline
   winner.

## Acceptance criteria

- Boolean-mask enumeration and integer cardinalities agree exactly.
- A hard-rejected candidate cannot be selected.
- A non-tied baseline ranking is unchanged.
- Equal extension masses preserve stable baseline order.
- A real tie selects the largest extension mass.
- Disabled or unavailable extension data does not change baseline selection.
- Malformed or over-limit input raises a typed error.

## Kill criteria

Reject the mechanism if it uses output length as a proxy, changes non-tied
rankings, accepts malformed masks, exceeds the finite resource limit, uses a
post-decision observation, or has no causal caller that can provide
pre-decision continuation sets.

CUDA component verification is software/invariant evidence only. It is not
evidence of generalisation or benchmark intelligence.
