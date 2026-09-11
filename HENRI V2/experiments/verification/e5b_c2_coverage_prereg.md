# E5b — C2 Token-Stream Coverage Gate (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-11-E5B-C2-COVERAGE
**Carrier:** `carrier/e5-wave-superposition` (base `10f5f23`; E5a promotion `815a778`)
**Supersedes:** the E5a coverage gate on C1 (`E5A_COVERAGE_FAIL`, seal `#1387`)

## Why C2

E5a measured coverage on the **C1 sentence-window** construct and found the target
too diffuse for any bounded zero-trainable candidate set (no generator reached
γ = 0.90 even at k = 2048; `const@k=1 = 0.440` exactly reproduced E4a's marginal,
validating the split). The C1 marginal is dominated by the sentence-final period.
The **C2 token-stream** construct has marginal **0.117** and a frozen-backbone
oracle of **0.431** — real headroom for a bounded set.

## Construct and split

C2 token-stream: prefix = the token window of length CTX = 128 ending at position
`i`; gold = the token at position `i`. Fresh regions, disjoint from every prior
carrier (E4a C1 windows 11000–21999; E4c C2 positions 300000–320000 and
600000–601000):

- calibration: token positions **700,000 – 719,999**
- evaluation: token positions **800,000 – 800,999**

Label: `CONDITIONAL_FRESH_SPLIT_SAME_CORPUS` +
`CONDITIONAL_CONTAMINATION_UNDISCLOSED` (contamination scan still pending).

## Frozen artifacts

| Artifact | Pin |
|---|---|
| corpus `wikitext2_train.parquet` | sha256 prefix `e83889ba` |
| tokenizer `tokenizer.json` | sha256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| backbone `Qwen/Qwen2.5-0.5B` | rev `060db6499f32faf8b98477b0a26969ef7d8b9987`, shard sha `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| E4b codec | `e4b_position_codec.PositionBoundCodec`, 0 trainable |

## Arms (all zero-trainable; backbone is eval-only)

| Arm | Generator | Role |
|---|---|---|
| **C_const** | top-k most frequent calib gold tokens | constant floor |
| **C_ctx** | top-k gold tokens observed after the context's last **word** (calib only) | surface/context floor |
| **C_wave** | the same library keyed on the terminal word **decoded from the context wave** by the sealed E4b position channel | the mechanism |
| **C_backbone** | frozen Qwen2.5-0.5B top-k at the last context position | §2-sanctioned ceiling |

The wave arm supplies the conditioning variable **itself**; the gap to C_ctx is
the decode tax. C_backbone is the ceiling: if it cannot reach γ at a bounded k,
no candidate-set interface is admissible on this construct.

**Attribution rule:** the backbone arm is a **ceiling**, never a competitor. No
HENRI capability may be claimed from it.

## Bounded k (no unbounded gate)

`k ∈ {1, 5, 16, 64, 256, 1024}`. An unbounded "reaches γ at some large k" gate is
rejected as nearly vacuous: a constant set of all calib golds covers everything
at k = |golds|. The reported endpoints are coverage at **k ≤ 64** and the coverage
**cost** `k90` (the k needed to reach 0.90).

## Pre-registered gates

- **G-C2-A (admissibility, PRIMARY):** C_backbone reaches γ = 0.90 at k ≤ 64.
- **G-C2-B (wave utility):** C_wave ≥ C_ctx − 0.02 at k = 64.
- **G-C2-C (cross-check, DIAGNOSTIC):** C_backbone top-1 within ±0.05 of 0.431.
- **G-C2-D (floor sanity):** `const@k=1` within ±0.02 of the C2 marginal ≈ 0.117;
  if not, `BLOCKED_INFRA` (split or gold construction wrong).
- **KILL:** if G-C2-A fails → `E5B_CONSTRUCT_INADMISSIBLE`. If G-C2-D fails →
  `BLOCKED_INFRA`. Rank metrics are **WITHHELD** whenever G-C2-A fails.

## Implementation requirements (bug classes already measured this session)

1. **Decode reference:** compare the decoded terminal word against
   `tokenize(prefix)[-1]` — the **prefix's own** terminal word, never the window's
   last word. (This exact bug produced a false "decode acc 0.001" in E5a v2 and a
   false 0/96 in E4b v1.)
2. **Library:** key on the context's last word -> observed **gold token ids** from
   calib windows only. Do not build it from within-prefix bigrams; those never
   observe a continuation (E5a v1 defect: coverage ~10x low).
3. **Types:** the library holds token **ids**; never pass them through a
   word-expansion helper (E5a v3 `TypeError`).
4. **Decode cost:** vectorized argmax over a precomputed `e:`-cell matrix, one
   op per sample. An O(|vocab|) Python loop is infeasible.

## Metric-C correction (prerequisite, user-directed)

`Sagnacfunctor.txt`'s `compute_sagnac_delta = max(0, 1 − Re<p,a>/D)` is measured
**VACUOUS** (range 3e-05 at D = 65536: aligned 0.999985, orthogonal 1.000000,
anti 1.000015 — a dead memory passes it). Corrected to the sealed normalized form
`1 − Re<p,a>/(|p||a|)`, range [0, 2]. Receipt: `functor_metric_fix.json`.
No carrier gates on the defective formula.

## Verdicts

`E5B_GATES_PASS` / `E5B_CONSTRUCT_INADMISSIBLE` / `BLOCKED_INFRA`.
No promotion to main. No capability claim.


---

## AMENDMENT A1 (pre-seal, disclosed, 2026-09-11)

**Reason.** G-C2-D as written compared this carrier's `const@k=1` against a
C2 marginal of 0.117 quoted from a prior session summary. The authoritative
E4a receipt (`henri-telemetry/e3/e4a_construct_audit.json`, sha256 prefix
`0a1b86fe42d83136`) records C2 `marginal_baseline.p1 = 0.117` with pair split
calib `[0, 10000]` / eval `[84000, 85000]`.

**Finding.** Recomputed under that exact documented rule with PINNED matching
identifiers (corpus `e83889ba`, tokenizer `c0382117` — both equal to the E4a
receipt's own pins) the value is `p1 = 0.033`, not 0.117
(`e5b_identifier_recon.json`, sha256 prefix `7ee4564d`). The calib gold
UNIVERSE reproduces EXACTLY (`distinct_golds = 2435` == receipt), so the calib
region is confirmed identical; the eval-side figure is not reproducible.
The cross-region reference is therefore `UNREPRODUCIBLE_REFERENCE` and is
**not used as a gate**.

**Amendment.** G-C2-D is re-expressed as a SAME-REGION floor-stability check:
the calib top-1 token's frequency on calib must be within +/-0.02 of its
frequency on the eval region of the same split.

**Scope limit.** NO other gate, threshold, gamma (0.90), k-list, arm, or
verdict rule is changed. The PRIMARY admissibility gate G-C2-A keeps its
pre-registered `k <= 64` budget; it is NOT relaxed.

**Hashes.** prereg before `02920cd9f6e57268756512d3f6423eef`.

**Conditioning-variable note.** On C2 the 128-token window ends mid-word
(19/100 golds are subword continuations of the tail), so a WORD-keyed
successor library cannot represent the target. v2 keys the library on the
token n-gram with pure backoff (`tok1` -> `tok2` -> const) and reports the
word-keyed wave arm alongside for comparability.
