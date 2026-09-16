# K-A Amendment A1 — the original K-A gate is VACUOUS; replace the criterion

**Status:** `PRE-REGISTERED BEFORE A1 EXECUTION`
**Supersedes:** K-A as written in `SPEC-2026-09-16-AAII-V43-RD-PIPELINE.json` §6.4
**Date:** 2026-09-16 | **Commit measured:** `7750abba6f21803fe80355de3a390c0e80784d06`

## 1. What K-A measured

Probe: `HENRI V2/experiments/verification/kaa_egress_probe.py`
Run: `kaa-egress-probe__7750abba6f21__20260916T174313Z-d162d38a`
Checkpoint: `LOADED`, sha256 `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060`,
799,034,119 bytes. Device `cpu` (`cuda_available=false`). 512 item rows written.

| Arm | distinct top-1 | entropy_mean (nats) |
|---|---|---|
| A `ring_uint8_scaled` (codec Z_256 → real) | **39 / 128** | 9.9858 |
| B `transduce_real` (transducer → real) | **19 / 128** | 9.9892 |
| C `random_control` | **37 / 128** | 9.9838 |
| D `shuffle_control` | 19 / 128 | — |
| uniform reference `ln(32000)` | — | 10.3735 |

## 2. Two defects in the K-A design (both disclosed)

**D1 — the gate is vacuous.** The pre-registered criterion was
"if distinct top-1 tokens ≤ 1 across N≥100 distinct prompts → `FALSIFIED_NO_EGRESS`".
Content arms scored 39 and 19. But the **negative control** — pure
`torch.randn` waves — scored **37/128**, essentially the same as the best content
arm. A gate that its own negative control passes cannot detect the claimed
mechanism. The original K-A verdict (`EGRESS_DISCRIMINATES`) is therefore
**withdrawn** as `GATE_VACUOUS`.

**D2 — `D_shuffle_control` is vacuous by construction.** It permutes the order of
the *same* prompt set. A permutation cannot change the *set* of top-1 tokens, so
the control is guaranteed to reproduce arm B's count. It tests order-independence
only. It carries no information about content sensitivity and must not be cited
as a control.

## 3. Root cause of the vacuity

`distinct_top1` measures **how many different tokens appear**, not **whether the
token depends on the input**. In a 32,000-way softmax with ~10 nats of entropy, a
content-blind head still yields many distinct argmax values across 128 draws.
Distinctness is therefore a **false-positive generator** for this mechanism.

The correct axis is **differential response**: does a *semantically local*
change move the representation *less* than a *distant* change? That is a paired
comparison inside the same head, which cancels the head's arbitrary but fixed
token geometry.

## 4. Amendment A1 — the replacement criterion

Measure near/far pair discrimination **at two boundaries** so that an ingress
defect and an egress defect are separable:

- **Boundary W (wave):** cosine similarity between encoded waves.
- **Boundary L (logit):** cosine similarity between output logits.

Pairs (64 prompts → 32 near, 32 far, seed fixed):

- **near pair:** same template with one content word changed
  (`What is the capital of topic-007?` vs `What is the country of topic-007?`).
- **far pair:** different template AND different content topic.

Controls:

- **Noise control:** 128 random waves → 32 near / 32 far by the same rule
  (near = small angular perturbation, far = independent draw). The noise margin
  is the floor.
- **Determinism check:** the same prompt encoded twice must give a bit-identical
  logit vector and identical top-1. A failure here is `ERROR_NONDETERMINISTIC`.

### Verdict rule (pre-registered, fixed thresholds)

Let `M_W = mean(near_cos_W) − mean(far_cos_W)` and `M_L` likewise.

| Condition | Verdict |
|---|---|
| determinism fails | `ERROR_NONDETERMINISTIC` |
| `M_W ≤ 0` (CI includes 0) | `INGRESS_LOSES_CONTENT` — the encoder is the defect |
| `M_W > 0` and `M_L > 0` and `M_L` CI lb > noise margin | `EGRESS_CARRIES_CONTENT` |
| `M_W > 0` and (`M_L ≤ 0` or `M_L ≤ noise margin`) | `EGRESS_DESTROYS_CONTENT` — the head is the defect |
| `M_W ≤ noise margin` | `INGRESS_WEAK` — encoder margin at noise floor |

**Effect size gate:** `M_L` must also exceed the noise-control margin. A margin
that a random-wave pair also achieves is not evidence (the D1 lesson).

## 5. Consequences for the pipeline

- S1/S2/S3 harness building may proceed: they measure **code execution**, which
  does not depend on the open-answer egress path.
- The **open-answer** members (AA-Omniscience, GDP.pdf, AA-LCR, HLE) stay
  `BLOCKED_PENDING_A1`. If A1 returns `EGRESS_DESTROYS_CONTENT`, the sanctioned
  egress gateway (phase-subtraction + Hopfield lexical snap + CEGIS + SGLD) is
  the required carrier, and the bare linear head is confirmed non-sanctioned —
  which the corpus already asserts (`INFERRED`) and this would make `OBSERVED`.
- No score claim of any kind follows from A1. It is a mechanism verdict on a
  CPU probe.

## 6. Session-level lesson (recorded)

Two of the four K-A arms were vacuous — one because the negative control matched
the treatment, one because the control was a permutation that cannot change the
measured set. **A control must be able to fail.** Before sealing any gate, ask:
*can the negative control pass this?* If yes, the gate measures nothing.
