# Phase 8.39 — Trained-decoder ranking head verdict

Status: **FALSIFIED (pre-registered kill fired: solved < 3/50)**
Branch: `phase839/humaneval-wave-ast` @ `8135372`
Date: 2026-08-20

## Result (OBSERVED, CUDA RTX 5090, n=50, d=65,536)

| Arm | solved | expressible | infra_errors | reordered |
|---|---|---|---|---|
| Control (no flag, same SHA) | 2/50 | 50 | 0 | 0 |
| `--decoder-rank` | **0/50** | 50 | 0 | **48** |

Checkpoint: `henri_decoder_checkpoint.pt`, SHA-256 `75572389083455a3…`, strict load PASS, 4 keys (down_proj 2048×65536, lm_head 32000×2048).

## Oracle probe (OBSERVED, CUDA)

Correct solutions ranked by decoder entropy scorer:

| Item | correct body | enum rank | decoder rank | n_cands |
|---|---|---|---|---|
| HumanEval/23 | `return len(string)` | 3 | **49** | 71 |
| HumanEval/35 | `return max(l)` | 5 | **68** | 71 |

## Mechanism (DERIVED)

The trained unbinder was trained on qFHRR wave-token pairs; its low-entropy peak is "single-token-like" geometry. Full-program superposition waves land in low-density regions → flat logits → high entropy. Entropy ranking therefore de-ranks program-shaped candidates — the learned prior transfers NEGATIVELY to full-program waves. 48/50 items reordered proves the lever engaged; the oracle proves the sign is wrong.

## Pre-registered gate

Kill: solved >= 3/50 → PASS. OBSERVED 0/50 → **KILL FIRED**. `--decoder-rank` stays default-OFF, unpromoted.

## Disposition

- Learned-prior lever class CLOSED for the code path: exemplar transfer (1/50) and decoder entropy (0/50) both falsified with mechanism evidence.
- Honest current external score: **HumanEval 2/50** (enumeration-order baseline), sealed at `79222a7`/`0066e9a`.
- Next levers are representation changes requiring offline training (MBPP cross-benchmark ranker, program-wave unbinder retraining) — philosophy-conflicting, not executed without approval.

## Evidence artifacts

- Scorecard: `experiments/verification/humaneval_decoderrank_839_scorecard.json` (item-level).
- Oracle probe: live CUDA python (ranks above).
- Log: `/root/telemetry_logs/humaneval_decoder_rank_839.log`.
- Governance event: appended (event store).
