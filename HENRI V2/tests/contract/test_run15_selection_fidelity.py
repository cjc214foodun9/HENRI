"""Contract tests for the run15 selection-fidelity mechanisms.

1. Structural-complexity penalty: the decoder's ranking must penalize
   off-manifold residual energy when a manifold projector (the R-EDMD
   low-rank basis V) is supplied, reordering candidates by complexity.
2. Rank-aware CEGIS escalation: with escalate=True, a true solution
   displaced below the primary attempt window is still found.
"""

import torch

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec


def _make_codec():
    return qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")


def _wave(codec, text: str) -> torch.Tensor:
    return torch.nn.functional.normalize(
        (codec.encode_text(text).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1),
        p=2, dim=0)


def test_complexity_penalty_reorders_by_residual():
    """The penalty term must order low-residual over high-residual candidates
    even when the raw similarity is nearly equal (lambda=0 ties)."""
    from mbpp_wave_ast_decoder import WaveASTDecoder
    codec = _make_codec()
    dec = WaveASTDecoder(codec, device="cpu")
    D = 1024
    base = _wave(codec, "def solve(x):\n    return sum(x)")
    # V: orthonormal basis whose first column IS base (the task manifold
    # contains the target). low is exactly in-span; high is off-span with a
    # comparable raw similarity to base.
    g = torch.Generator().manual_seed(7)
    filler = torch.nn.functional.normalize(torch.randn(D, 15, generator=g), p=2, dim=0)
    filler = filler - (base @ filler) * base.unsqueeze(1)  # orthogonalize vs base
    V = torch.nn.functional.normalize(torch.cat([base.unsqueeze(1), filler], dim=1), p=2, dim=0)
    low = base  # in-span: residual ~ 0
    w = torch.nn.functional.normalize(torch.randn(D, generator=g) - torch.dot(torch.randn(D, generator=g) * 0, base) * base, p=2, dim=0)
    w = w - torch.dot(w, base) * base
    w = torch.nn.functional.normalize(w, p=2, dim=0)
    high = torch.nn.functional.normalize(base + 0.5 * w, p=2, dim=0)

    def score_of(v):
        coeffs = v @ V
        resid = torch.norm(v - coeffs @ V.t()) / torch.sqrt(torch.tensor(float(D)))
        return resid.item()

    resid_low, resid_high = score_of(low), score_of(high)
    # dimension-normalized residuals (L2 / sqrt(D)): high must carry
    # measurable off-manifold energy and strictly exceed low.
    assert resid_high > 0.005, f"high residual too small: {resid_high}"
    assert resid_high > resid_low, f"residuals not separated: {resid_low} vs {resid_high}"
    sim_low = float(torch.dot(low, base).item())
    sim_high = float(torch.dot(high, base).item())
    assert abs(sim_low - sim_high) < 0.15, f"raw sims not comparable: {sim_low} vs {sim_high}"
    lam = 0.5
    s_low = sim_low - lam * resid_low
    s_high = sim_high - lam * resid_high
    assert s_low > s_high, f"penalty failed to reorder: {s_low} vs {s_high}"


def test_decoder_penalty_keeps_simple_target_top():
    """With a manifold V fit on simple solution waves, the decoder ranks the
    simple target above the multi-statement distractors."""
    from mbpp_wave_ast_decoder import WaveASTDecoder
    codec = _make_codec()
    D = 1024
    g = torch.Generator().manual_seed(11)
    simple_sources = [
        "def solve(x):\n    return sorted(x)",
        "def solve(x):\n    return sum(x)",
        "def solve(a):\n    return a[::-1]",
        "def solve(a):\n    return len(a)",
    ]
    V = torch.nn.functional.normalize(
        torch.stack([_wave(codec, s) for s in simple_sources]).t().float(), p=2, dim=0)
    # full-rank stack -> re-fit r=16 via SVD for a proper orthonormal basis
    U, S, _ = torch.linalg.svd(torch.stack([_wave(codec, s) for s in simple_sources]).t().float())
    V = U[:, :16]
    dec = WaveASTDecoder(codec, device="cpu")
    target = "def solve(x):\n    return sum(x)"
    pred = _wave(codec, target)
    pw = _wave(codec, "Write a function to sum a list.")
    cands = dec.decode(pred, pw, "solve", ["x"], manifold_proj=V, complexity_lambda=0.2)
    sources = [c[0] for c in cands]
    assert target in sources
    rank = sources.index(target)
    assert rank < 5, f"simple target displaced to rank {rank}"
    # the recursion body (multi-statement, high residual) must rank below the
    # simple target
    rec = "def solve(x):\n    if x <= 1:\n        return x\n    return x * solve(x - 1)"
    if rec in sources:
        assert rank < sources.index(rec), "multi-statement distractor above simple target"


def test_cegis_escalation_finds_displaced_solution():
    from mbpp_cegis_synthesizer import MbppCegisSynthesizer
    from types import SimpleNamespace

    class _FakeSandbox:
        def __init__(self, passing_src):
            self.passing_src = passing_src

        def execute(self, src):
            ok = self.passing_src in src
            return SimpleNamespace(status="PASS" if ok else "FAIL")

    ex = [{"task_id": 1, "code": "def solve(x):\n    return sum(x)"}]
    codec = _make_codec()
    synth = MbppCegisSynthesizer(ex, codec, device="cpu")
    true_src = "def solve(x):\n    return sum(x)"
    item = {"test_list": ["assert solve([1, 2, 3]) == 6"]}
    ranked = [(f"def solve(x):\n    return sorted(x)", {}, 0.9),
              (f"def solve(x):\n    return set(x)", {}, 0.8)] * 6
    ranked.insert(13, (true_src, {}, 0.7))
    # without escalation: miss
    code, meta = synth.cegis_verify(ranked, item, _FakeSandbox(true_src), max_attempts=12, escalate=False)
    assert code is None
    # with escalation: the true solution at index 13 is found
    code, meta = synth.cegis_verify(ranked, item, _FakeSandbox(true_src), max_attempts=12, escalate=True)
    assert code == true_src
    assert meta["candidates_tried"] == 13
    assert meta["cegis_escalated"] is True
