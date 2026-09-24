# UHR-05 — HENRI's code egress is bounded at 48 characters: the SciCode wiring question is answered

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Evidence classes:** `OBSERVED` · `DERIVED` · `FALSIFIED`

## The question

The previous sprint closed with this pre-registered next step: *"swap the backbone for
HENRI's own wave→text egress on the same 48-item SciCode window."* Before building it,
the egress's actual output contract had to be measured — and it had to be measured with a
**real** wave, not a synthetic one.

## What I first got wrong

I recorded, and then corrected, two opposite errors in the same session. Both are kept
here because each was load-bearing:

| Claim | Verdict | Evidence |
|---|---|---|
| "The egress is a `1-of-V` selector, not a generator → wiring it into SciCode is a CATEGORY ERROR" | **`FALSIFIED`** | `henri_decoder.py:391` `decode_autoregressive_sequence` is a REAL multi-token loop (`:415 for step in range(max_tokens)`, `:421 generated_token_ids.append`); `.pt` checkpoint exists |
| "The egress can emit code, so SciCode is constructible" | **`FALSIFIED`** | the code branch emits ONLY from a 10-entry stub vocabulary (measured below) |

The first error came from reading one class (`HoloEgressCodebook`) and generalising to the
repo. The second came from reading the *control flow* (`:680` routes `"def "` prompts to
the code branch) without measuring the *emittable set*. **A branch that exists is not a
capacity.**

## The measurement

```
A. code_vocab_map — the egress's ENTIRE emittable token set for the code branch
     entries                        = 10
     total characters if ALL emitted = 48
     longest single token            = 12 chars

B. the requirement
     SciCode 10.1 ground_truth_code  = 943 chars (365 lines)

C. decode with a REAL wave
     codec  = zone_c_epistemic_axiom_harness.qFHRREpistemicCodec   (the module
              execute_authentic_coding_benchmark.py:30 itself imports)
     goal   = codec.bind_hadamard(codec.encode_text("SCICODE_CODING_OPERATOR"),
                                  codec.encode_text(<prompt>))
     result = DecoderEgressFailClosedError: out-of-vocab token id 23247

D. decode with a RANDOM wave (control)
     result = DecoderEgressFailClosedError: out-of-vocab token id 29674
```

**`DERIVED`:** the code egress has a hard ceiling of **48 characters**. A SciCode sub-step
needs **943**. So a run wired to this path yields `pass@1 = 0/48` **by construction**, and
that zero would carry **no information about HENRI's intelligence** — it is a property of
the emittable set, not of the model.

**`OBSERVED`:** both waves fail closed with a *typed* error rather than emitting a stub.
`henri_decoder.py:424-428` says why: *"Out-of-vocab token id must fail closed — never
modulo-alias into a hand-written stub map (template-skeleton trap)."* That is correct
engineering and it is why this measurement is possible at all: a system that aliased
silently would have produced plausible-looking code and a fake score.

## Why the ceiling exists (and is not a bug)

```
HENRINeuralEgressUnbinder.__init__ :77-81
   down_proj = nn.Linear(65536, 2048, bias=False)
   lm_head   = nn.Linear(2048, 32000, bias=False)     <- the FULL 32000-token head EXISTS
checkpoint : schema henri.decoder-checkpoint.v1-legacy, d_model 65536, vocab 32000
   state_dict sha256 = 72cf3ab5a18b2f9ece83107e…       load_status = LOADED
PhaseRingCodebookDecoder.decode_autoregressive_sequence :412-428
   code_vocab_map = grammar_masker.code_vocab_map     <- 10 entries, 48 chars
   masked_logits  = grammar_masker.mask_logits_for_step(logits, token_strings, step)
```

The unbinder projects onto all 32 000 tokens; the **grammar mask** then restricts the code
branch to a 10-token stub set. The mask is the binding constraint, not the checkpoint.
Widening the vocabulary is therefore a *design decision on `HENRI_ASTGrammarMask`*, not a
model retraining problem — and it must be validated against
`grammar_masker.is_valid_ast` (`:439`, `:443`), which already refuses to emit an invalid
AST.

## What this means for the AAII path

| Assessment | Status |
|---|---|
| Wiring the current code egress into SciCode | **`FALSIFIED` as a capability measurement** — bounded at 48 chars |
| The egress fails closed rather than fabricating | `OBSERVED` — correct; no synthetic score possible |
| The 799 MB decoder checkpoint loads and validates | `OBSERVED` — `LOADED` / `TRAINED_DECODER` |
| SciCode harness ready to receive a real generator | `OBSERVED` — committed `87efc9707067`: window 48, reference arm 48/48, official targets 50/50 |
| Any AAII v4.3 score | `BLOCKED` — 75 % of index weight is externally graded |

## Next falsification (re-scoped by this measurement)

The question is no longer "does HENRI's egress beat a 1.5 B backbone" — it cannot, through
this path. It is:

1. **What is the intended production emitter for code?** `code_vocab_map` is a stub set by
   construction. `humaneval_wave_ast_runner.py` uses `qfhrr_ast_discriminative_kernel`
   instead — that is the candidate real path, and it must be measured the same way
   (emittable-set size first, decode second).
2. **If the mask is the constraint, quantify it:** `|code_vocab_map|`, the per-step
   allowed set, and the maximum AST-complete string reachable at `max_tokens=32`. A
   production channel needs ≥ 943 chars for one SciCode step.
3. **Only then** a paired run on the committed 48-item window, against the 5/48 backbone
   baseline, with the same instrument and pin.
