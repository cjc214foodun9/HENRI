# Gate 3.1 Verdict — P@1 / P@5 / Perplexity diagnostic (measured FAIL)

**Carrier:** carrier/p1-pk-diagnostics @ `01a4399` (prereg sha256 `53f97d952dc53e574ce742ba…`, sealed `#20c33745`)
**Run:** vast-5090 GPU-exclusive, clean detached worktree @ `01a4399`, single run, no retries

## Measured values (OBSERVED, from /root/gate31/gate31_receipt.json)

| Metric | Measured | Pre-registered bound | Result |
|---|---|---|---|
| P@1 | 0.016 | ≥ 0.285 | FAIL |
| P@5 | 0.033 | ≥ 0.640 | FAIL |
| Perplexity (teacher-anchored single-shot NLL) | 146622.9446 | reported only | — |
| Random-control P@1 | 0.0 | scale reference | — |
| n_eval | 1000 | 1000 | OK |

- Checkpoint sha256 prefix `08747c70…` verified; corpus sha256 prefix `e83889ba…` verified; teacher rev `060db6499f32…`; seed 20260908; label `CONDITIONAL_SAME_CORPUS_HELDOUT`; definition `teacher-anchored single-shot NLL (non-autoregressive)`.
- Receipt: `/root/gate31/gate31_receipt.json` (sha256 `9309d57837a256da…`), pulled to `henri-telemetry/gate31/gate31_receipt.json`.

## Verdict: `GATE31_FAIL`

The calibrated E2 egress head does not meet the user-supplied token-level
P@1/P@5 bounds on the sealed 1,000-window evaluation split. E2_GATES_PASS
(`#3714dcdb`) stands for its own gates (isometry, loss descent, k-NN alignment
margin, rotation consumption); token-level retrieval accuracy is a separate,
failed measurement. No threshold renegotiation within this carrier; no
checkpoint mutation; no retraining.

## Honest status

- Gate 3.4 (sealed gauntlet dispatch) is **NOT executed**: the user's own
  protocol requires Action 1 (engagement) AND Action 2 (P@1/P@k) to pass before
  main is touched. Action 2 failed; Action 1 still shows zero progress events.
- Main remains `10f5f23` (untouched).
