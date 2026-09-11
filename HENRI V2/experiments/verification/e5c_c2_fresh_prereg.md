# E5c — Verified-Fresh C2 Re-Run (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-11-E5C-C2-FRESH
**Carrier:** `carrier/e5-wave-superposition` (base `10f5f23`; E5b `64372543`)
**Relationship to E5b:** E5b's verdict `E5B_CONSTRUCT_INADMISSIBLE` STANDS. E5c asks
a different question and does NOT re-litigate admissibility.

## Question

E5b measured, on a region later disclosed as possibly E4c-contaminated,
`wave@64 = 0.399` vs `tok1@64 = 0.459` (deficit −0.060). Is that deficit
**region-specific** (an artifact of the contested region) or **construct-scoped**
(a property of a word-keyed library on token streams)?

## Disjointness (fail-closed, prerequisite)

`e5_fresh_regions.json` is the sole authority for the split. It is compiled from
SOURCE FILES read this session (file:line recorded in the receipt), not from any
summary. Consumed ranges: E4a C2 calib/eval, E4c oracle-probe C2 eval
(`e4c_oracle_probe.py:53`), E4c probe2 C2 eval (`e4c_probe2.py:51`), E5b v1, E5b
v2, plus a **computed** C1 window token span for windows 11000..21999.

Chosen region **R1** (20,000-token calibration + 1,000-token evaluation), asserted
disjoint from every consumed range with margin ≥ 10,000 tokens. The carrier
**refuses to run** unless the prover receipt exists with
`VERDICT == FRESH_REGIONS_PROVEN` and R1 matches its `chosen.R1`.

> **FALSIFIED-ELSEWHERE NOTE.** A region table claiming E4c consumed
> `(300k,320k) (600k,620k) (700k,720k) / (600k,601k) (800k,801k) (900k,901k)` at
> `e4c_factorial.py:24-34` is **not used**: direct read of that file shows lines
> 24–34 are imports (`from pathlib import Path` … `import torch.nn.functional as F`).
> No such block exists there.

## Frozen artifacts

| Artifact | Pin |
|---|---|
| corpus `wikitext2_train.parquet` | sha256 `e83889ba…` |
| tokenizer `tokenizer.json` | sha256 `c0382117…` |
| backbone `Qwen/Qwen2.5-0.5B` | shard sha `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| E4b codec | `e4b_position_codec.PositionBoundCodec`, 0 trainable |
| prover receipt | `e5_fresh_regions.json` (fail-closed gate) |

## Arms (all zero-trainable; backbone eval-only)

| Arm | Generator | Role |
|---|---|---|
| C_const | top-k calib gold tokens | constant floor |
| C_tok1 | top-k golds after the previous TOKEN (calib only) | surface floor |
| **C_wave** | E4b terminal-word channel, exact E5b-v2 configuration | comparability arm |
| C_rand | deterministic random set | negative control |
| C_backbone | frozen Qwen2.5-0.5B top-k at the last context position | **ceiling, never a competitor** |

## Gates

- **G-E5C-A (PRIMARY):** `wave@64 ≥ tok1@64 − 0.02` → the deficit was
  **region-specific** (PASS). Otherwise **construct-scoped** (FAIL).
- **G-E5C-B (diagnostic):** backbone top-1 within ±0.05 of the E4a oracle 0.433.
- **G-E5C-C:** same-region floor stability: calib top-1 token frequency vs its
  eval-region frequency, ≤ 0.02.
- **Report-only:** backbone@64 vs γ = 0.90 (context; E5b already failed it).

The cross-region marginal 0.117 is **NOT** a gate — it is recorded as
`UNREPRODUCIBLE_REFERENCE` (`e5b_identifier_recon.json`, `e5b_index_map.json`).
No gate may reference a different region's baseline.

## Verdicts

`E5C_DEFICIT_REGION_SPECIFIC` / `E5C_DEFICIT_CONSTRUCT_SCOPED` / `BLOCKED_INFRA`.

Diagnostic carrier. No positive C2 claim. No promotion. No capability claim.
Rank metrics withheld if G-E5C-A fails. main stays `10f5f23`.
