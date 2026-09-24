# UHR-05 — No HENRI code path can emit a SciCode solution: both are bounded, measured

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Evidence classes:** `OBSERVED` · `DERIVED` · `FALSIFIED`

## The question this closes

Commit `0a684a1` measured the code egress in `henri_decoder.py` (10-token stub
vocabulary, 48 characters maximum) and named the open item:

> *"What is the intended production emitter for code? `code_vocab_map` is a stub set by
> construction. `humaneval_wave_ast_runner.py` uses `qfhrr_ast_discriminative_kernel`
> instead — that is the candidate real path, and it must be measured the same way
> (emittable-set size first, decode second)."*

That measurement is now done, and the answer is that the candidate real path is **also**
bounded.

## The trace

```
humaneval_wave_ast_runner.py:42   from wave_ast_decoder import WaveASTDecoder
                           :107   decoder = WaveASTDecoder(codec, device=device)
                           :273   candidates = decoder.decode(prompt_wave, target_wave, entry, args)
wave_ast_decoder.py:169           class WaveASTDecoder
                  :181            def _instantiate(self, entry, args) -> list[str]
                  :238            def decode(self, pred_wave, prompt_wave, entry, args, ...)
```

`decode` **does not generate**. Its own code:

```
:244  "Enumerate the grammar under the item signature, rank every complete program by
       transformation-relative wave similarity MINUS a structural-complexity penalty"
:262  for body in self._instantiate(entry, args):        <- HAND-WRITTEN TEMPLATES
:263      src = f"def {entry}({', '.join(args)}):\n{body}"
:265      ast.parse(src)                                  <- syntax gate
:286  scored.sort(key=lambda t: t[2], reverse=True)       <- WAVE RERANK
```

`_instantiate` (`:181-236`) fills bodies from four hand-written template families —
`_ifelse_bodies` (`:116`), `_loop_bodies` (`:127`), `_recursive_bodies` (`:151`),
`_index_bodies` (`:160`) — plus fixed expression lists (`EXPRS_UNARY`, `EXPRS_BINARY`,
`NARY_3`, `NARY_4`). The wave is used to **order** those templates, never to author one.
The runner's own comment agrees: *"The candidate SET is unchanged; only attempt order
moves."* (`humaneval_wave_ast_runner.py:280`)

## The measurement

```
signature                    nargs  bodies  parseable  longest body
get_alpha(recvec, alpha_...)      2     172        172      106 chars
is_palindrome(text)               1      71         71       95 chars
g(a,b,c)                          3      10         10       26 chars
h(a,b,c,d)                        4       4          4       26 chars
k(a,b,c,d,e)                      5       4          4       29 chars
m(a,b,c,d,e,f)                    6       4          4       32 chars
f()                               0       0          0        0 chars
```

**Against the SciCode 10.1 requirement:**

| | value |
|---|---|
| required signature | `get_alpha(recvec, alpha_scaling)` |
| `WaveASTDecoder` emittable bodies | **172** (all parse) |
| longest such body | **106 chars** |
| SciCode 10.1 ground truth | **943 chars** / 482 lines |

`DERIVED`: the maximum body this decoder can produce for the required signature is
**106 characters against a 943-character target** — an 8.9× shortfall, and the bodies are
single template statements (`return ...`, a loop, an if/else), not multi-statement
numerical routines. A run through this path yields `pass@1 = 0/48` **by construction**.

Note that 0-argument signatures produce **zero** candidates, which the runner already
classifies `NOT_EXPRESSIBLE` (`:252`) rather than `FAIL`, consistent with its docstring at
`:12` (*"Items whose signature the grammar cannot express (0 args, >=5 args with …)"*).

## So both HENRI code paths are bounded — by design, not by defect

| path | mechanism | ceiling | SciCode needs |
|---|---|---|---|
| `PhaseRingCodebookDecoder.decode_autoregressive_sequence` (`henri_decoder.py:391`) | 10-token stub grammar mask (`code_vocab_map`) | **48 chars** | 943 chars |
| `WaveASTDecoder.decode` (`wave_ast_decoder.py:238`) | template enumeration + wave rerank | **106 chars** for a 2-arg signature | 943 chars |

Both fail CLOSED with typed errors (`DecoderEgressFailClosedError`, `NOT_EXPRESSIBLE`)
rather than emitting plausible-looking stubs. That is correct engineering: a system that
aliased or fabricated would have produced a fake score instead of this measurement.

## What this means for the AAII path

**This is a representational-capacity result, not a wiring result.** Wiring either path
into the committed 48-item SciCode harness (`87efc9707067`) would produce a `0/48` that
is a property of the emittable set, and would carry **no information about HENRI's
intelligence**. Repeating the earlier mistake of running it anyway — as I nearly did with
the backbone at a denominator of 2 — would waste a run and invite a false
`FALSIFIED_AT_SCAFFOLD` conclusion about the model.

The honest statement: HENRI's present egress is a **selection and reranking** architecture
over hand-authored candidates. Producing a 943-character numerical routine requires a
generator with that capacity, which does not exist in these modules.

## Evidence status

| Claim | Class | Basis |
|---|---|---|
| `WaveASTDecoder` enumerates templates and reranks | `OBSERVED` | `:262`, `:263`, `:286` read; runner comment `:280` |
| 172 bodies / 106 chars for the required signature | `OBSERVED` | executed `_instantiate` for the real signature |
| 0-arg signature yields 0 candidates | `OBSERVED` | executed |
| SciCode 10.1 needs 943 chars | `OBSERVED` | corpus `ground_truth_code` |
| "`humaneval_wave_ast_runner` is the real emitter" | **`FALSIFIED`** | it is a reranker over a fixed candidate set |
| Either path yields a HENRI capability number | **`FALSIFIED`** | bounded at 48 / 106 chars vs 943 |
| Any AAII v4.3 score | `BLOCKED` | 75 % of index weight externally graded |

## Next falsification

Do **not** run either egress on SciCode. Instead:

1. **Search for a generator with ≥ 943-char capacity** across the tree
   (`decode_autoregressive_sequence`, `_instantiate`, and `generate_text` are the three
   found; the transformer-backed `henri_backbone_adapter.generate_text` is the only one
   with unbounded output, and it is the 5/48 baseline already measured).
2. **If none exists, the finding stands:** HENRI's coding channel is a reranker, and the
   AAII-relevant work is the *generator* — not the harness, which is now validated
   (`reference arm 48/48`, official targets `50/50`, window 48).
3. **Report the capacity bound as the session's coding result**, with the two ceilings
   (48 / 106 chars) as the falsifiable measurement.
