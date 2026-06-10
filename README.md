# HENRI Loop-Closure Patches — Phases 0, 1, 3

Four files. All math numerically validated before delivery.

| File | Phase | Replaces |
|---|---|---|
| `sagnac_veto.py` | 1 | the tautological interferometer |
| `ground_truth.py` | 0 | the wave-distance pass/fail decision |
| `henri_contract.py` | 3 | scattered magic dimensions + half-wave DB truncation |
| `verify_integration.py` | wiring | the body of `verify` / `execute_reasoning_loop`'s decision |

---

## Phase 1 — `sagnac_veto.py` (the tautology, fully resolved)

The old two lines collapsed `transmission_truth` to `target_manifold`
identically. The replacement is an orthogonal projection (the balanced
coherent homodyne decomposition your manuscript already adopted):

```
c     = <t_hat, psi>          # homodyne coefficient
truth = c * t_hat             # constructive / transmitted component
delta = psi - truth           # destructive / reflected residual
E     = ||delta||^2           # genuine misalignment, 0..||psi||^2
```

Verified properties: `truth != target` for arbitrary candidates;
`<truth,delta>=0` (exact); `||psi||^2 = ||truth||^2 + ||delta||^2`;
`E=0` iff parallel; `E=||psi||^2` iff orthogonal.

**Drop-in:** identical `forward(wave, target) -> (truth, delta, error_energy)`
signature, so `zone_b_emulator.py` and `cognitive_swarm.py` need no call-site
changes. Just replace the file.

---

## Phase 0 — `ground_truth.py` (install the authority)

`GroundTruthGate` runs the candidate against the task's examples and returns a
graded reward.

- `evaluate_arc(code, examples, repl)` — binds each example's `input` as
  `INPUT_GRID`, runs the candidate, parses the final printed line as the
  predicted grid, scores cell accuracy. `passed` requires every example exact.
- `evaluate_tests(code, tests, repl)` — `{"call","expect"}` dicts or bare
  assertion strings.

Reuses your existing `universal_repl.execute_block` — no second sandbox.

**Candidate contract:** the model's program must `print(json.dumps(answer))`
as its final stdout line. Add one line to the swarm system prompt in
`swarm_registry.py` stating this.

---

## Phase 3 — `henri_contract.py` (unify + lossless DB)

- `DIMS` / `HenriDims`: import dimensions from here everywhere. Replace the
  scattered `gemma_dim=2048`, `hrr_dim=4096`, `rank=16`, `boundary_dim=64`.
  At boot, after probing the model: `DIMS = DIMS.with_measured_gemma(measured)`
  once, then pass that object down.
- `complex_to_db(psi, hrr_dim)` / `db_to_complex(vec, hrr_dim)`: lossless
  codec. Replace **every** occurrence of the `db_vec[:2048]` packing in
  `cognitive_swarm.py`, `lookahead_prefetcher.py`, and `init_local_db_4096.py`.
- The column must become `vector(8192)`. Run `migrate_schema_sql()` or
  `fresh_schema_sql()`. **Existing rows are corrupt (50% discarded) and must be
  re-seeded — do not decode them.**
- `enforce_shape(name, tensor, shape)`: call at every LoRA/centroid load site
  in place of the silent reinit-on-mismatch. Hard failure surfaces lost
  weights instead of discarding them.

---

## Wiring — `verify_integration.py`

`CorrectedVerifier.corrected_verify(...)` shows the composition. Copy its body
into your verifier and **delete** the old decision path. The authority is now
`gt.passed`; the interferometer `error_energy`/`alignment` become *secondary*
signals for centroid drift and telemetry only.

This sets up Phase 2 cleanly: the `reward` from the gate is the objective the
LoRA/centroid updates should optimize, and `_truth_wave`/`_delta_wave` anchor
on the candidate — not the target.

## Validation

Every claim above was checked numerically:
- interferometer: 6/6 properties pass
- gate: correct / partial / wrong / broken / unit-test cases all correct
- contract: bit-exact round trip (0.0 error), old-bug 50%-loss demonstrated,
  strict rejections fire
- integration: objective gate passes a correct solution even with a misaligned
  wave (proving the authority moved off the wave)
