# Carrier G3 — Wave-Packet Path Search: Results

**Directive:** user approval (2026-09-01) + `holographic search.pdf`
(`HENRI-AUDIT-2026-09-V3-QUANTUM-WAVE-SEARCH`, 190,418 B, SHA `76c28f6b69f1…`, ledger @1,162).
**Prereg:** `docs/spec/g3_wave_packet_path_search_preregistration.md` (SHA `4a2e39caa6ea…`, sealed `#df349291` @1,167).
**Branch:** `carrier/g3-wave-packet-search` @ `6430f4c`.

## Verdict

**`G3_SIDECAR_VERIFIED` — diagnostic sidecar passes C1–C8 on the RTX 5090 CUDA target (D=65,536). No capability or benchmark claim.**

| Gate | Result |
|---|---|
| C1 default-OFF | PASS (env-gated `HENRI_G3_WAVE_PACKET`; production runner never imports) |
| C2 frozen determinism | PASS (same-seed byte-identical generators; **0 trainable params**) |
| C3 norm preservation | PASS (norm_err **0.0** at D=65,536) |
| C4 veto selectivity | PASS (aligned path dominates; HENRI Sagnac delta) |
| C5 top-k bounded | PASS (≤64 paths; best = argmax coherence) |
| C6 complexify adapter | PASS (norm-preserving, injective, third-family sidecar) |
| C7 no-policy influence | PASS (static: runner source free of module/flag) |
| C8 latency (diagnostic) | 0.1565 s for H=8 × 7-actions at D=65,536 CUDA; finite clearance |

## Audited corrections (implemented, tested)

1. **Frozen deterministic generators** (zero `nn.Parameter`) — PDF's `nn.Parameter` rejected under the zero-trainable invariant.
2. **HENRI normalized Sagnac delta** `1 − Re⟨a,b⟩/(‖a‖‖b‖) ∈ [0,2]` — PDF's `sin²` formula rejected.
3. **Complex flat `[D]` = third wave family** — one-way norm-preserving complexification adapter, diagnostic sidecar only, no policy influence.

## Evidence

- Receipt: `C:/Users/chan/AppData/Local/Temp/g3_receipt.json` (SHA `0191c8be8d6e…`)
- Remote: `/tmp/henri_g3_wave_packet/g3_receipt.json`; worktree `/tmp/g3-wave-packet-wt` @ `6430f4c` (clean, 11/11 CUDA)
- Engine: `HENRI V2/experiments/verification/arc_g3_wave_packet_search.py`; tests `test_g3_wave_packet_search.py` (11/11 local + remote CUDA)
- Regression: 1,067 passed / 6 skipped (G3 +11, no regressions)

## Next (NOT authorized)

Planner wiring of `WavePacketPathSearch` into the live action path = separate approval-gated carrier with its own prereg (replaces/benchmarks against `EFEPlanner.select_action`; PDF's MCTS-replacement claim is stale — MCTS is not the live planner).
