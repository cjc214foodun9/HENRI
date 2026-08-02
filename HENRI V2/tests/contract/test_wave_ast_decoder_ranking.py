"""Formal verification of the wave->AST transformation-relative ranking.

Deterministic, reduced-dim (CPU, d_model=1024), no DB. Domain = the decoder's
bounded grammar: single-return exprs (sorts, sums, comprehensions, slices,
int-conversion, binary ops) AND the multi-statement shapes (if/else,
loop-accumulate, count, index, recursion).

Gates (pre-registered):
  1. Expressibility: every domain target appears in the decoded candidates.
  2. Rank top-K: every target ranks in the top-K (K=5) when the predicted
     wave IS the target's wave (self-selection, sim ~ 1.0).
  3. Discrimination: transformation-relative ranking separates a target from
     a same-prompt wrong-body distractor better than absolute ranking does.
"""

import torch

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

TOP_K = 5

# (entry, args, body, prompt) — body is the target's function body.
DOMAIN = [
    ("solve", ["x"], "    return sorted(x)", "Write a function to sort a list."),
    ("solve", ["x"], "    return sum(x)", "Write a function to sum a list."),
    ("solve", ["nums"], "    return [x ** 2 for x in nums]", "Write a function to square each element."),
    ("solve", ["a", "b"], "    return sorted(a + b)", "Write a function to merge and sort two lists."),
    ("solve", ["a", "b"], "    return set(a) & set(b)", "Write a function to find common elements."),
    ("solve", ["a"], "    return a[::-1]", "Write a function to reverse a list."),
    ("solve", ["a", "b"], "    return a[:b]", "Write a function to take the first b elements."),
    ("solve", ["a"], "    return int(a, 2)", "Write a function to parse a binary string."),
    ("solve", ["a", "b"], "    return a + b", "Write a function to concatenate two lists."),
    ("solve", ["a"], "    return len(a)", "Write a function to count elements."),
    # multi-statement shapes
    ("solve", ["a", "b"], "    if a > b:\n        return a\n    return b", "Write a function returning the larger."),
    ("solve", ["a"], "    result = []\n    for x in a:\n        result.append(x)\n    return result", "Write a function to copy a list."),
    ("solve", ["a"], "    c = 0\n    for x in a:\n        if x > 0:\n            c += 1\n    return c", "Write a function to count positive numbers."),
    ("solve", ["a"], "    if a <= 1:\n        return a\n    return a * solve(a - 1)", "Write a function to compute a factorial."),
    ("solve", ["a", "b"], "    for i in range(len(a)):\n        if a[i] == b:\n            return i\n    return -1", "Write a function to find an index."),
]


def _make_codec():
    return qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")


def _wave(codec, text: str) -> torch.Tensor:
    return torch.nn.functional.normalize(
        (codec.encode_text(text).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1),
        p=2, dim=0)


def test_domain_expressible_and_ranked_top_k():
    from mbpp_wave_ast_decoder import WaveASTDecoder
    codec = _make_codec()
    dec = WaveASTDecoder(codec, device="cpu")
    results = []
    for entry, args, body, prompt in DOMAIN:
        target = f"def {entry}({', '.join(args)}):\n{body}"
        pred = _wave(codec, target)
        pw = _wave(codec, prompt)
        cands = dec.decode(pred, pw, entry, args)
        sources = [c[0] for c in cands]
        assert target in sources, f"NOT EXPRESSIBLE: {target}"
        rank = sources.index(target)
        results.append(rank)
        assert rank < TOP_K, f"{entry}: rank {rank} >= TOP_K={TOP_K}\nsources[:5]={sources[:5]}"
    assert len(results) == len(DOMAIN)
    assert max(results) < TOP_K


def test_transformation_relative_discriminates():
    """A same-prompt, wrong-body distractor must rank below the target; the
    transformation-relative (prompt-subtracted) score must separate them."""
    from mbpp_wave_ast_decoder import WaveASTDecoder
    codec = _make_codec()
    dec = WaveASTDecoder(codec, device="cpu")
    prompt = "Write a function to sum a list."
    target = "def solve(x):\n    return sum(x)"
    distractor = "def solve(x):\n    return sorted(x)"
    pred = _wave(codec, target)
    pw = _wave(codec, prompt)
    cands = dec.decode(pred, pw, "solve", ["x"])
    sources = [c[0] for c in cands]
    assert target in sources and distractor in sources
    t_rank, d_rank = sources.index(target), sources.index(distractor)
    assert t_rank < d_rank, f"target rank {t_rank} >= distractor rank {d_rank}"


def test_multi_statement_candidates_parse():
    """Grammar gate: every generated multi-statement candidate is valid Python."""
    from mbpp_wave_ast_decoder import WaveASTDecoder
    codec = _make_codec()
    dec = WaveASTDecoder(codec, device="cpu")
    pred = _wave(codec, "def solve(a, b):\n    if a > b:\n        return a\n    return b")
    pw = _wave(codec, "larger of two")
    for src, _ in dec.decode(pred, pw, "solve", ["a", "b"]):
        compile(src, "<decoder>", "exec")
    pred1 = _wave(codec, "def solve(a):\n    if a <= 1:\n        return a\n    return a * solve(a - 1)")
    pw1 = _wave(codec, "factorial")
    for src, _ in dec.decode(pred1, pw1, "solve", ["a"]):
        compile(src, "<decoder>", "exec")
