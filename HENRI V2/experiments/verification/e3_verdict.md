# E3 Verdict — Egress-Head Reform (measured FAIL, both arms)

**Carrier:** carrier/e3-egress-reform @ `778cf34` (base `5d4a399` = p1pk tip)
**Prereg:** `e3_egress_reform_prereg.md` sha256 `26d38d8380868fab64db357521b86f2a…`, sealed `#8049aec4`
**Run:** vast-5090 GPU-exclusive, clean detached worktree @ `778cf34f` (WT_DIRTY=0),
overlay-free path, artifact preflight PASS (`ckpt 08747c70…`, `te 48b174e9…`,
`corpus e83889ba…`), scaffold RC=0, full RC=1, single run, no retries.

## Measured (OBSERVED, `/root/e3/full/e3_receipt.json`, sha256 `f738ff7adfad2d6c22d56176…`)

| Arm | P@1 | P@5 | Perplexity | Bound (0.285 / 0.640) | Verdict |
|---|---|---|---|---|---|
| A — probe-distribution readout (zero training, E2 ckpt `08747c70…`) | **0.015** | **0.029** | 141730.3618 | FAIL | **E3A_FAIL** |
| B — generative CE head (E2 warm start, frozen tied teacher readout) | **0.079** | **0.139** | 142539.0327 | FAIL | **E3B_FAIL** |

Arm B gates: G1 isometry **3.41e-06** (≤1e-4) PASS · G2 CE descent **11.9346 → 11.6916**
(needs < 0.9×first = 10.741) **FAIL** · G4 rotation consumption 10/16 (≥8) PASS.
**Export: NOT written** (`e3_egress_production.pt` absent on remote) — fail-closed per prereg.

## What this falsifies (the important part)

1. **The "construct mismatch" diagnosis is FALSIFIED.** If Gate 3.1's failure were a
   readout-construct defect (raw centroid vs token-level distribution), Arm A — the
   same frozen E2 checkpoint read through its own k-NN softmax probe (k=16, τ=0.07)
   → cluster centroid → token scores — should have recovered the gap. It measured
   **P@1 0.015 vs Gate 3.1's 0.016**: statistically indistinguishable. Re-reading the
   same features through the mechanism's own probe does not change the outcome.
2. **The residual gap is supervision/capacity, not readout.** Arm B trained token-level
   CE on the sealed 10,000-window calibration split and moved P@1 from 0.016 → 0.079
   (≈4.9×) — real but far below the 0.285 bound. CE began at 11.9346 against
   `ln(151936) = 11.9312` (uniform) and ended at 11.6916: a **0.24-nat** movement in
   3 epochs. Learning a 151,936-way token distribution from 10k windows through a
   frozen tied readout is supervision-starved by roughly the vocabulary/pair ratio.
3. **Warm start + frozen tied readout is a binding constraint, not a convenience.**
   The adapter must move wave features into alignment with *all* 151,936 token
   directions while the readout itself cannot adapt.

## Labels and limits

- `CONDITIONAL_SAME_CORPUS_HELDOUT` — eval = sealed E2 partition (1,000 windows,
  disjoint from the 10,000 calibration windows, same corpus). The split has now been
  measured twice (Gate 3.1 raw-point readout; E3 Arms A and B) — disclosed. No
  fresh-domain generalization claim.
- Teacher-anchored single-shot NLL; **not** autoregressive LM perplexity.
- Bounds (0.285 / 0.640) were **not** renegotiated; no retries; no post-hoc tuning;
  no threshold adjustment after observing results.
- **No promotion. No main change.** Main remains `10f5f23`.
- Receipt: `/root/e3/full/e3_receipt.json`, pulled to
  `henri-telemetry/e3/e3_receipt.json` (local sha `f738ff7adfad2d6c22d56176…`).

## Honest next options (each requires a NEW prereg + approval)

1. **Supervision scale** — build pairs from the full 36,718-row corpus (not the first
   10k windows), pre-registering the pair count; the 4.9× movement suggests CE is
   learning, just slowly.
2. **Untied trainable readout** — train a `[d_target → V]` head (or `lm_head` itself)
   with CE, dropping the frozen tied constraint; report parameter count and
   non-trainable buffers separately.
3. **Change the contract** — accept that token-level P@1/P@5 bounds require a
   revision-pinned pretrained backbone, with contamination review and matched
   ablations (a contract change, not a tuning change).
