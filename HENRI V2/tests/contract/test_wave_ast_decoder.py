"""Contract tests for the wave->AST structural decoder (mbpp_wave_ast_decoder).

Local, deterministic, reduced-dim (CPU, d_model=1024). The DB-free gates:

  1. Grammar validity: every decoder candidate parses as Python.
  2. Structure recovery: for a synthetic single-return solution, the EXACT
     solution must rank in the decoder's own top-K (DECODE_TOP_K=5).
  3. Non-autoregressive slot scoring: slot scoring uses only the predicted
     wave; candidates are assembled via ast and verified parseable.
"""

import torch

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec


def _make_codec():
    return qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")


def test_decoder_candidates_all_parse():
    from mbpp_wave_ast_decoder import WaveASTDecoder, DECODE_TOP_K
    codec = _make_codec()
    dec = WaveASTDecoder(codec, device="cpu")
    pred = torch.nn.functional.normalize(
        (codec.encode_text("def solve(x): return sorted(x)").to(torch.float32) /
         (codec.k_bins - 1) * 2.0 - 1.0).view(-1), p=2, dim=0)
    prompt = torch.nn.functional.normalize(
        (codec.encode_text("Write a function to sort a list.").to(torch.float32) /
         (codec.k_bins - 1) * 2.0 - 1.0).view(-1), p=2, dim=0)
    cands = dec.decode(pred, prompt, "solve", ["x"])
    assert len(cands) > 0
    for src, meta in cands:
        compile(src, "<decoder>", "exec")  # grammar gate: all candidates parse


def test_decoder_ranks_exact_solution_top_k():
    from mbpp_wave_ast_decoder import WaveASTDecoder, DECODE_TOP_K
    codec = _make_codec()
    dec = WaveASTDecoder(codec, device="cpu")
    target = "def solve(x):\n    return sorted(x)"
    prompt = "Write a function to sort a list."
    pred = torch.nn.functional.normalize(
        (codec.encode_text(target).to(torch.float32) /
         (codec.k_bins - 1) * 2.0 - 1.0).view(-1), p=2, dim=0)
    prompt_wave = torch.nn.functional.normalize(
        (codec.encode_text(prompt).to(torch.float32) /
         (codec.k_bins - 1) * 2.0 - 1.0).view(-1), p=2, dim=0)
    cands = dec.decode(pred, prompt_wave, "solve", ["x"])
    sources = [c[0] for c in cands]
    # The exact source must appear in the decoded candidate list (the decoder
    # must be able to EXPRESS the solution) and rank in the top-K.
    assert target in sources, "decoder cannot express the target solution"
    rank = sources.index(target)
    assert rank < DECODE_TOP_K, f"exact solution ranked {rank} (top-K={DECODE_TOP_K})"


def test_decoder_signature_rename():
    from mbpp_wave_ast_decoder import WaveASTDecoder
    codec = _make_codec()
    dec = WaveASTDecoder(codec, device="cpu")
    pred = torch.nn.functional.normalize(
        (codec.encode_text("def k_smallest(list1, n): return sorted(list1)[:n]").to(torch.float32) /
         (codec.k_bins - 1) * 2.0 - 1.0).view(-1), p=2, dim=0)
    prompt = torch.nn.functional.normalize(
        (codec.encode_text("k smallest").to(torch.float32) /
         (codec.k_bins - 1) * 2.0 - 1.0).view(-1), p=2, dim=0)
    cands = dec.decode(pred, prompt, "k_smallest", ["a0", "a1"])
    for src, _ in cands:
        assert src.startswith("def k_smallest(a0, a1):"), src
