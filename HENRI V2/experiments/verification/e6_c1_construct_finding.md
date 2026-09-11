# E6a — C1 Construct Finding (decisive, CPU, 6.6 s)

**Receipt:** `e6_c1_construct.json` sha256 `60153536d4503599`
**Verdict:** `C1_CONSTRUCT_DEGENERACY_CONFIRMED_AND_FRESH_REGION_PROVEN`
**Status:** SEALED. This is a diagnostic; it produces no carrier.

## 1. Instrument trust passed exactly

| quantity | this walk | reference | dev |
|---|---|---|---|
| const@k=1 | 0.44 | 0.44 | 0 |
| const@k=64 | 0.721 | 0.721 | 0 |
| const@k=2048 | 0.839 | 0.839 | 0 |
| max abs deviation vs E5a | **0.0** | — | — |
| dot token rate (calib) | 0.4269 | 0.4269 (E4a) | 0 |
| distinct golds (calib) | 1890 | 1890 (E4a) | 0 |

A diagnostic whose instrument cannot reproduce the reference measurement cannot
diagnose anything. This one reproduces it to six decimals.

## 2. H1 (construct defect) is SUPPORTED

Top C1 golds by calib frequency:

| id | count | rate | token |
|---|---|---|---|
| 659 | 4269 | **0.4269** | `' .'` |
| 284 | 687 | 0.0687 | `' ='` |
| 1154 | 239 | 0.0239 | `' ,'` |
| 279 | 228 | 0.0228 | `' the'` |
| 220 | 210 | 0.0210 | `' '` |

- period token rate **0.4269**; punctuation rate **0.5703**
- the period alone is **97.0 %** of the k=1 marginal (0.4400)
- so the C1 marginal is effectively **one token**

Consequence: a constant generator emitting `.` scores 0.44 at k=1
and 0.721 at k=64 — which is why both conditional arms
(ctx 0.318, wave 0.332) sit
**below** the constant arm. The E5a deficit was not a wave-channel failure.

## 3. Repairing the gold does NOT rescue the gate

Content-only construct (alphanumeric golds, 0.429 of eval retained):

| k | 1 | 5 | 16 | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| const | 0.083916 | 0.202797 | 0.27972 | 0.375291 | 0.48951 | 0.561772 | 0.62704 |

marginal falls 0.4400 -> 0.083916, but
const@2048 reaches only 0.62704 and
k90 is **None** inside K_LIST.
Removing the punctuation scaffolding spreads the marginal; it does not make a
bounded zero-trainable candidate set admissible.

## 4. A second, independent limiter: key coverage

| quantity | value |
|---|---|
| calib distinct terminal words | 4012 |
| eval terminal word present in calib | **0.679** |

Roughly a third of eval conditioning keys are unseen. Any keyed library is capped
at ~0.68 before the wave channel is consulted. This is a construct/config limiter,
separate from the gold degeneracy.

**Key-definition caveat (disclosed):** this script keys on the raw whitespace
terminal word; E5a keyed on `tokenize()` (lowercase, non-alnum stripped). The
measured ctx gap (0.371 vs
0.318) is therefore a key-definition difference, NOT an
improvement. It is not claimed as a gain.

## 5. Stationarity — why a fresh-region C1 re-run is predicted to fail

| quantity | consumed region (E4a/E5a) | fresh region |
|---|---|---|
| dot token rate | 0.4269 | 0.4399 |
| marginal k=1 | 0.4400 | 0.466 |
| const@64 | 0.7210 | 0.739 |
| distinct golds | 1890 | 1785 |

The construct is **stationary across the corpus**. The degeneracy is a property of
the gold definition (*first BPE token of the sentence's last word*), not of the
region. A fresh-region C1 re-run is therefore **pre-registered to fail
identically**, and running it would spend GPU time to confirm a result already
decided here on CPU in 6.6 seconds.

## 6. Fresh region proven anyway (reusable asset)

- window index: calib `[56014, 66014]`, eval `[66014, 67014]`
- token span: `[1_650_209, 1_963_038]`
- margin 10,000; **0 violations** against all 13 source-anchored consumed ranges

This region is available to any future construct that is NOT C1-as-defined.

## 7. Evidence labels

| claim | class |
|---|---|
| walk reproduces E5a const@k, max dev 0.000000 | `OBSERVED` |
| dot rate 0.4269 / punct 0.5703 / period = 97 % of marginal | `OBSERVED` |
| content-only const@2048 = 0.627, k90 None | `OBSERVED` |
| key coverage 0.679 | `OBSERVED` |
| fresh region disjoint, 0 violations | `OBSERVED` |
| C1 construct is stationary corpus-wide | `DERIVED` (3 quantities agree across regions) |
| a fresh-region C1 re-run will fail identically | `HYPOTHESIS` — falsifiable, but the cheap kill is already done |
| any HENRI capability gain | `NOT_EVALUATED` |
