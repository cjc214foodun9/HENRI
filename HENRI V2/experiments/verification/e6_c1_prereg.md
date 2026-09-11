# E6b — C1 Carrier Preregister (REPAIRED construct only)

**Spec:** HENRI-SPEC-2026-09-12-E6B-C1-REPAIRED
**Carrier:** `carrier/e6-physical-verifier`
**Status:** SPEC. Not run this session.

## Why this is not the originally-approved C1 carrier

The approved item was "C1 word-boundary construct (decode 0.839, non-degenerate
conditioning)". The second half of that premise is **falsified** by
`e6_c1_construct.json` sha `60153536d4503599`:

- C1 gold is **0.4269** the period token; the period is
  **97 %** of the k=1 marginal.
- Conditioning is therefore degenerate by construction, and **both conditional
  arms score below the constant arm** (ctx 0.318,
  wave 0.332, const 0.721).
- A fresh region does not help: the construct is stationary
  (fresh marginal 0.466 vs 0.4400;
  fresh const@64 0.739 vs 0.7210).

The decode 0.839 figure is **valid** and is preserved: it measures terminal-word
recovery, which is a wave-channel property and is unaffected by gold degeneracy.
What is withdrawn is the claim that this construct can host a bounded
zero-trainable candidate set.

## Required repairs before any run (both, not either)

1. **Gold definition.** Exclude punctuation-only and whitespace-only golds
   (content-only construct). Measured effect: marginal 0.4400 -> 0.0839,
   const@2048 -> 0.627, k90 -> None. This repair alone is **insufficient**.
2. **Key coverage.** The terminal-word key must cover the eval inventory. With
   whitespace keys, only **0.679** of
   eval keys appear in calib. Candidate repairs, each requiring its own
   measurement:
   - key on the last **k** words (back-off), which is the shape the E4a
     `prefix_last3_backoff` baseline already uses;
   - key on a **character n-gram** of the terminal word to absorb OOV;
   - key on the **token tail** (E5d showed this is a bijective recoding of the
     token id on C2; on C1 the tail is a *sub-word* of a word, which is a
     different object and must be measured, not assumed).

**Gating rule:** no E6b run is admissible until (a) the repaired construct's
`const@k<=64` is measured, and (b) key coverage is reported. If neither repair
lifts const@64 toward 0.90, the bounded-candidate-set interface is inadmissible on
C1 as well, and the interface should be retired for token-prediction constructs
generally rather than repaired a third time.

## Floor specification (learned from E5c/E5d)

The binding floor is **`max(const, tok1, tok2, ..., tokN)`**, NOT `tok1`.
E5c/E5d measured `const@64 > tok1@64` on three prover-verified disjoint regions;
a `tok1`-only floor credits conditional arms falsely. Any E6b gate compares
against the maximum over all zero-trainable arms.

## Instrument trust gate (mandatory, before any other number is reported)

`const@k=1` must reproduce E4a's C1 marginal **0.440 within +/-0.02**, and the full
`const@k` curve must reproduce E5a's table within **0.005**. E6a met this to
0.000000; E6b must re-earn it on its own region.

## Fresh region

Reserved for E6b (proven disjoint, margin 10,000, 0 violations):
window index calib `[56014, 66014]`, eval `[66014, 67014]`,
token span `[1_650_209, 1_963_038]`.

**Note:** this region's *C1-as-originally-defined* statistics are already measured
(dot rate 0.4399, marginal
0.466). It must NOT be consumed by a re-run of
the broken construct. It is reserved for the repaired construct.

## Runtime assertions (designs out the E5b compliance defect)

The E5b defect was that the *executed* split differed from the *declared* split.
E6b asserts, at runtime, that the executed region constants equal the constants in
this prereg's text, and fails closed on mismatch. The region is parsed from this
file, not duplicated in code.

## Kill criteria

- `const@k<=64 < 0.90` after both repairs -> the bounded candidate-set interface
  is INADMISSIBLE on C1. Record and stop. Do not attempt a third repair.
- key coverage still `< 0.85` after repair -> the conditioning variable is
  inadequate; the defect is in the key, not the channel.
- `const@k` curve deviates `> 0.005` from E5a -> instrument untrusted; no verdict.

## Evidence labels

- C1 degeneracy, repairs' measured effect, fresh region: `OBSERVED`
- key-repair candidates: `HYPOTHESIS`
- E6b outcome: `NOT_EVALUATED`
