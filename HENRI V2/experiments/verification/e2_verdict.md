# E2 Verdict — k-NN softmax egress calibration (measured PASS)

**Carrier:** carrier/e2-egress-knn-scale @ `f441f6b6` (base c8fc94a0 = E1 tip)
**Prereg (sealed):** e2_egress_knn_prereg.md, final sha256 `e58e6b41…` (seal `#24aa4e04`, reconcile `#564a99a1`)
**Run:** remote verify (Vast 5090, GPU-exclusive, exact SHA `f441f6b6`, overlay `7557238908`, DSN `set -a`)

## Preconditions (OBSERVED)
- SHA_PREFIX_OK f441f6b6; WT_DIRTY=0; HAS_DSN=True
- FULLSUITE_RC=0; CONTRACT_RC=0 (E2 8/8 + E1 7/7)
- Corpus: Salesforce/wikitext-2-raw-v1 @ rev `b08601e0…` → parquet 36,718 rows, col `text`,
  SHA-256 `e83889ba…`, 6,357,543 B (matches local re-read)
- Teacher: Qwen/Qwen2.5-0.5B rev `060db6499f…`, teacher_embeddings.pt [151936, 896],
  tokenizer.json 7,031,645 B; shard pin `88c14255…` (verified on E1 carrier previously)

## Scaffold (≤16 windows)
SCAFFOLD_RC=0 — 16 calib windows, loss 3.466→descent; receipt `/root/e2-calib/out_scaffold/scaffold_receipt.json` (309 B)

## Full scale (10,000 calib / 1,000 eval)
`[E2] corpus_rows=36718 sentences=85758; pairs calib=10000 eval=1000; steps=939`

| Gate | Criterion (prereg) | Measured | Verdict |
|---|---|---|---|
| G1 isometry | ≤1e-4 | 1.317e-11 | PASS |
| G2 descent | last-mean < 0.9×first-mean | 3.0072 → 0.8952 (loss 3.466→0.596) | PASS |
| G3 retrieval | align_trained − max(untrained, random) ≥ +0.05 | trained 0.3577, untrained −0.0745, random −0.0738 → margin +0.432 | PASS |
| G4 causal | ≥8/16 probes | 16/16 | PASS |

**Verdict: `E2_GATES_PASS`**

## Export (per prereg §5.4 only — no promotion, no default flip)
`/root/e2-calib/out_full/e2_egress_production.pt` — 614,545,228 B
SHA-256 `08747c70c462f7592c85ec363558b87ec74d6f226fd1adfb5420106f04a42fa2` (verified on remote)

## Disclosure (honest limits)
- P@1 / P@k diagnostics from prereg §4 were NOT emitted by the runner (runner gap; gate G3
  is margin-based per prereg and was evaluated). Not re-run: gate criterion met, diagnostics
  are reporting-only. No capability claim beyond the measured egress-calibration gates.
- No external benchmark (AAII v4.2 / ARC-AGI-3) was dispatched; all 10 AAII members remain
  NOT_EVALUATED. Stage 3 (default flip, production coupling) is REQUIRES_APPROVAL and does
  NOT follow from this carrier.
- Main untouched: 10f5f23 (verified via git ls-remote).

## Governance
`#3714dcdbc6e21a52` E2_GATES_PASS (henri_audit.py record). Telemetry:
`C:/Users/chan/henri-telemetry/e2/{scaffold,full}_receipt.json`; this doc + receipts committed
to `carrier/e2-egress-knn-scale`.
