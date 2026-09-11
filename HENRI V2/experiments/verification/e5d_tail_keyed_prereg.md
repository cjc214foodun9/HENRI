# E5d — Token-Tail-Keyed Wave Variant (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-11-E5D-TAIL-KEYED
**Carrier:** `carrier/e5-wave-superposition` — separate prereg and separate seal
from E5c (user directive: its own carrier).

## Motivation (measured, not assumed)

On the C2 token-stream construct, **19/100 golds are subword CONTINUATIONS of the
128-token window tail** (`e5b_marginal_diag.json`, sha `569573b3`). A successor
library keyed on the last **whitespace word** cannot represent such targets: E5b
measured wave decode accuracy 0.363–0.382 on C2, versus **0.839 on C1** where
windows end on word boundaries. The conditioning variable is the defect.

## Repair under test

Key the successor library on the terminal **TOKEN PIECE** rather than the
whitespace word, and decode that piece from the E4b position channel.

The codec reuses the **validated E4b math byte-for-byte**:
`_ring` (SHA-256-seeded Z_256 ring, D=2048), `_pos_cells("e:"+key)` (16 cells),
position-parity signs, energy = `mean(sign(wave[cell]))`. The **only** change is
segmentation: pinned-tokenizer pieces instead of `txt.split()`.

**Decode reference (the E5a-v2 bug class, explicitly guarded):** the decoded
terminal piece is compared against the **context's own terminal piece** — never
the gold token, and never the window's last whitespace word. That exact
reference mismatch produced a false `decode acc 0.001` in E5a v2.

## Disjointness (fail-closed, prerequisite)

Region **R2** from `e5_fresh_regions.json` — disjoint from every consumed range
with margin ≥ 10,000 tokens, distinct from E5c's R1. The carrier refuses to run
unless `VERDICT == FRESH_REGIONS_PROVEN` and R2 matches its `chosen.R2`.

## Arms (all zero-trainable)

| Arm | Generator | Role |
|---|---|---|
| C_const | top-k calib gold tokens | constant floor |
| C_tok1 | top-k golds after the previous **token** | surface floor |
| C_tok2 | token bigram with backoff to tok1 → const | stronger surface floor |
| **C_wavetail** | **piece-keyed E4b channel** | the repair (mechanism) |
| C_waveword | word-keyed E4b channel (E5b-v2 config) | negative control on the SAME split |
| C_rand | deterministic random set | negative control |
| C_backbone | frozen Qwen2.5-0.5B top-k | ceiling, never a competitor |

The word-keyed arm on the **identical split** is what makes the comparison
non-vacuous: it isolates the segmentation change as the single lever.

## Gates

- **G-E5D-A (PRIMARY):** piece-decode accuracy ≥ **0.80**.
- **G-E5D-B:** `wavetail@64 ≥ tok1@64 − 0.02` (decode cost bounded).
- **G-E5D-C (repair efficacy):** `wavetail@64 > waveword@64 + 0.05`.
- **G-E5D-D:** same-region floor stability ≤ 0.02.

## Verdicts

`E5D_REPAIR_CONFIRMED` (A∧B∧C∧D) / `E5D_REPAIR_INSUFFICIENT` / `BLOCKED_INFRA`.

Diagnostic carrier. No promotion. No capability claim. No training. main `10f5f23`.
