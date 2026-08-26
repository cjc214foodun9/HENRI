# Gate 3 — Grid-Egress Legal-Bounds Validation at the EFE-Planner Consumer (Preregistration, Carrier D)

Source: `HENRI_SOTA_Benchmark_Wiring_Evaluation.md` (PROPOSAL) Phase 3.
Prior disposition: Gate 3 `REQUIRES_DESIGN_CORRECTION`.
Branch: `feat/temporal-navigation-t0`. This document is the frozen contract.

## 1. Wiring-doc defect corrected

The proposal wired Gate 3 to `sagnac_mcts_planner.py`. The live action path is
`orch.plan_action -> EFEPlanner.select_action` with egress at the
`arc_egress_contract.decode_action_egress` consumer inside
`arc_curriculum_replay.py` step() (lines 429-432). The proposal's primary gate
`SyntaxError == 0.0%` is demoted to a DIAGNOSTIC: grid-token legality is a
necessary structural property, never a capability or task-efficiency score.
Lexical snap (`HENRI_LEXICAL_SNAP`) remains default-OFF; this carrier does not
wire or promote it.

## 2. Hypothesis (falsifiable)

The live EFE-planner egress path is structurally legal: every grid reaching
the decode boundary is within the encoder's dimensional and palette bounds
(dimensions <= max_grid_dim = 128, colors clamped to the legal palette [0, 15]
deterministically), and the egress contract fails closed (typed exception, no
silent fallback) on an unloaded transducer, an illegal wave shape, or an
invalid vocabulary.

## 3. Live anchors (source-verified)

- Tokenizer construction: `production_arc_run.py:721-725`
  `HENRIVisionEncoder(d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"], device=DEVICE, spatial_basis_kind=HENRI_ARC_SPATIAL_BASIS, bg_mask=HENRI_ARC_BG_MASK)`; SCALE = dict(d_model=65536, r_rank=16, num_blocks=8192); defaults `HENRI_ARC_SPATIAL_BASIS=default`, `HENRI_ARC_BG_MASK=0` (lines 475-477).
- Encoder behavior (`henri_vision_encoder.py`): `encode_spatial_grid` returns `[1, num_blocks, 8]`; asserts H, W <= max_grid_dim (128); clamps colors via `torch.clamp(grid, 0, 15)`; raises on fully-background grids.
- Egress consumer (`arc_curriculum_replay.py` step, lines 408-434): state wave from `encode_spatial_grid(grid).squeeze(0)`; chosen action decoded by `decode_action_egress(egress, predicted_wave, vocab, require_loaded=True)`; any exception -> `EGRESS_FAIL_CLOSED` in info, step returns None (fail-closed).
- Egress contract (`arc_egress_contract.py`): `flatten_uwe` requires exact `[num_blocks, 8]` and d_model numel else `EgressFailClosedError`; `ActionEgressVocabulary` rejects actions without `.name` and duplicates; `decode_action_egress` raises unless `checkpoint_load_status == "LOADED"` when `require_loaded=True`.

## 4. Validation procedure (frozen)

Execute `promotion_gate3_egress_validation.py` with the LIVE modules:

| Check | Procedure | Gate |
|---|---|---|
| G1 dimensional bounds | feed a grid with a dimension > 128 to `encode_spatial_grid` | must raise (ValueError/AssertionError) -> fail-closed confirmed |
| G2 palette legality | feed a grid containing color 99 (out of palette) | encoded wave deterministic; color domain clamped to [0, 15]; no NaN/non-finite at the decode boundary |
| G3 legal grid decode | encode a valid 8x8 grid (colors 0..9); shape check [1, 8192, 8], finite, non-zero | shape/finite PASS |
| G4 egress fail-closed: unloaded transducer | `decode_action_egress` with `require_loaded=True` on a stub with status != LOADED | raises `EgressFailClosedError` |
| G5 egress fail-closed: illegal wave shape | `flatten_uwe` on a [3, 5] tensor | raises `EgressFailClosedError` |
| G6 egress fail-closed: invalid vocabulary | `ActionEgressVocabulary` with a duplicate action or an action without `.name` | raises `EgressFailClosedError` |

Diagnostics (reported, not gates): OOB color frequency at the decode boundary
(expect 0 after clamping), malformed-grid token rate (expect 0.0 over the
fixture set), encoded wave row norms (finite), dimensions of the encoded wave.

## 5. Verdict precedence (fail-closed, total order)

```
BLOCKED_INFRA > FAIL_DIMENSIONAL_BOUNDS > FAIL_PALETTE_LEGALITY >
FAIL_EGRESS_FAIL_CLOSED > GATE3_VALIDATION_PASS
```

- NaN / harness error -> `BLOCKED_INFRA` (never scientific KILL for infra).
- G1 does not raise -> `FAIL_DIMENSIONAL_BOUNDS`.
- G2 wave non-finite or non-deterministic -> `FAIL_PALETTE_LEGALITY`.
- Any of G4-G6 does not raise -> `FAIL_EGRESS_FAIL_CLOSED`.
- All pass -> `GATE3_VALIDATION_PASS` (structural legality of the live path;
  NOT a capability, NOT a benchmark score, NOT a snap promotion).

## 6. Kill experiments (pre-registered)

1. G1 accepts an oversized grid -> dimensional bound is not enforced -> FAIL.
2. G2 produces non-finite or non-deterministic waves -> palette path broken -> FAIL.
3. G4/G5/G6 silently fall back instead of raising -> egress is not fail-closed -> FAIL.

## 7. Evidence labels

Metrics `OBSERVED` from the live Vast run of `promotion_gate3_egress_validation.py`
(torch CPU fixture, deterministic seed 20260826). Verdict per precedence.
`SyntaxError == 0` is reported as DIAGNOSTIC ONLY. Corpus consult:
`BLOCKED_AUTH` (NotebookLM stale, retried 2026-08-26) — retry before Gate 1 execution.

## 8. Artifacts

- This preregistration.
- `promotion_gate3_contract.json` (machine-readable contract).
- `promotion_gate3_egress_validation.py` (validation module, imports LIVE modules).
- `tests/contract/test_promotion_gate3_contract.py` (contract test).
- Receipt JSON written by the validation module.
