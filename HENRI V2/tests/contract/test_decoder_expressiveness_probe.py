"""Contract test for the --ast-decode expressiveness probe.

The probe must count exemplars whose OWN tests pass with the decoder's
candidates, proving the grammar + verification path are alive together.
"""

from types import SimpleNamespace

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec


class _FakeSandbox:
    """In-process executor mimicking SecurePythonSandbox.execute for CPU tests."""

    def execute(self, src):
        try:
            ns = {}
            exec(src, ns)
            return SimpleNamespace(status="PASS")
        except Exception:
            return SimpleNamespace(status="FAIL")


def test_probe_decoder_expressiveness_counts_passes():
    from mbpp_cegis_synthesizer import MbppCegisSynthesizer
    from mbpp_wave_ast_decoder import WaveASTDecoder
    codec = qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")
    # Two single-return exemplars the decoder grammar CAN express.
    exemplars = [
        {"task_id": 1, "text": "Write a function to sort a list.",
         "code": "def solve(x):\n    return sorted(x)",
         "test_list": ["assert solve([3, 1, 2]) == [1, 2, 3]"]},
        {"task_id": 2, "text": "Write a function to sum a list.",
         "code": "def solve(x):\n    return sum(x)",
         "test_list": ["assert solve([1, 2, 3]) == 6"]},
        # A solution the single-return grammar cannot express (DP-shaped).
        {"task_id": 3, "text": "Write a DP function.",
         "code": "def solve(x):\n    memo = {}\n    def f(i):\n        if i in memo:\n            return memo[i]\n        memo[i] = x[i]\n        return memo[i]\n    return f(0)",
         "test_list": ["assert solve([5]) == 5"]},
    ]
    synth = MbppCegisSynthesizer(exemplars, codec, device="cpu")
    dec = WaveASTDecoder(codec, device="cpu")
    probe = synth.probe_decoder_expressiveness(dec, exemplars, _FakeSandbox(), top_n=12)
    assert probe["total"] == 3
    assert probe["expressible"] >= 1, f"decoder+verification path dead: {probe}"
    assert probe["per_exemplar"][0] is True, "sorted exemplar must be expressible"
    assert probe["per_exemplar"][1] is True, "sum exemplar must be expressible"
