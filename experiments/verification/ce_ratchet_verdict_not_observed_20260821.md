# CE Ratchet — Verdict NOT_OBSERVED @ D=65,536 (CUDA)

- Packet: HENRI-CLASS47-CE-TELEMETRY-2026-08-21 (pre-registered, amended 4x pre-result)
- Device: cuda (RTX 5090, Vast 47411800), D=65,536, 8,192 blocks, 256 steps, seed 7
- Probe: `HENRI V2/scripts/verification/ce_ratchet_probe.py` (production EFEPlanner learner)
- Checkpoint overlay: `henri_decoder_checkpoint.pt` sha256 `7557238908...` (matches /root/henri-archive/manifest.sha256)
- Evidence: `experiments/verification/evidence/ce_ratchet_d65536_37c138a4.json` sha256 `37c138a4209bb947eab5ac10dd3ad0d6f1697204e29cfae59dd7c6b095a66785`

## Gates (pre-registered)

| Gate | Criterion | Measured | Verdict |
|---|---|---|---|
| T3 | ce_trained - ce_untrained > +0.01 | **-0.054991** | FAIL |
| T4 | ce_after_erase >= ce_untrained - 0.02 | 0.193943 vs 0.228643 | FAIL |
| S | |CE| <= 3, no NaN | 0.2486 / 0.1937 | PASS |
| Engagement | trained pred-cos > untrained; loss descends | 0.0271 > 0.0172; loss 1.083->1.020 | PASS |

## Verdict

**CE_RATCHET_NOT_OBSERVED@D65536** — sealed. Learning was provably engaged
(loss descends, pred-cos trained > untrained), so the negative deltas are a
genuine measurement of the production monolithic action-conditioned transition
operator at production scale, not a training-failure artifact and not
BLOCKED_INFRASTRUCTURE. Trained EI_macro dropped 0.405 -> 0.365 while EI_micro
stayed flat (0.1237 -> 0.1240): the CE loss is entirely macro-determinism loss.

## Masking hypothesis (INFERRED, corpus consult ca4bb787, 2026-08-21)

The bank names "Transition Model Fallacy #5 / action-conditioned transition
conflation": CE measured on an un-factored monolithic transition operator pools
distinct contexts under one TPM row -> high-entropy smeared row -> artificially
depressed macro EI -> false-negative CE. Corpus prescription: context-refined
TPM rows (split by before-state macro-cluster identity). This measurement does
not falsify the Levin ratchet globally; it falsifies the ratchet on the
current monolithic operator path. Next falsification (pre-registered before
run): context-refined TPM rows (CE computed per macro-cluster context, then
aggregated). Do NOT reopen the ratchet claim without that packet.

## Scope note

CE module stays default-OFF diagnostic (`--ce-telemetry`). No policy or score
path was touched by this measurement. Erasure arm: T4 ce_after_erase 0.1939
(trained) — erasing the window tail did not restore untrained-level CE.
