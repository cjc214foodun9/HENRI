"""Contract tests for the Phase C W_task goal channel (ARC closed-loop gate).

Reduced-dimension mechanics only (CPU, no checkpoint, no CUDA). The remote
A/B baseline run is the production evidence path. These tests verify:

1. W_task functor compilation + single-pass retrieval captures a consistent
   color-shift rule in Z_256 rings (healthy retrieval similarity).
2. The pre-registered kill gate: a consistent-rule functor exceeds the
   minimum similarity; an inconsistent/no-rule functor falls below it.
3. The gate logic used by production_arc_run.py (adopt goal only when
   retrieval similarity >= threshold).

Ring similarity uses the representation-aware circular distance
min(|a-b|, 256-|a-b|) * pi/128 -> cos, matching the C2b fix.
"""

import torch

from zone_c_epistemic_axiom_harness import HolographicTaskFunctorCompiler, qFHRREpistemicCodec


def _ring_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity over circular Z_256 phase distance."""
    d = torch.minimum(torch.abs(a.to(torch.int32) - b.to(torch.int32)),
                      256 - torch.abs(a.to(torch.int32) - b.to(torch.int32)))
    return float(torch.cos(d.float() * 3.141592653589793 / 128.0).mean().item())


def _rule_demo_pairs(n: int, d_model: int, shift: int = 3):
    """Synthetic demos: output rings = input rings + shift (mod 256)."""
    torch.manual_seed(7)
    xs = []
    ys = []
    for _ in range(n):
        x = torch.randint(0, 256, (d_model,), dtype=torch.uint8)
        y = ((x.to(torch.int32) + shift) % 256).to(torch.uint8)
        xs.append(x)
        ys.append(y)
    return list(zip(xs, ys))


def _random_demo_pairs(n: int, d_model: int):
    """No-rule control: independent random input/output rings."""
    torch.manual_seed(9)
    xs = [torch.randint(0, 256, (d_model,), dtype=torch.uint8) for _ in range(n)]
    ys = [torch.randint(0, 256, (d_model,), dtype=torch.uint8) for _ in range(n)]
    return list(zip(xs, ys))


def test_planner_no_all_zero_label_wart():
    """Regression guard: search() must not resurrect the C1-class inert
    all-zero bootstrap labels or the CE-only adapt_in_context call."""
    import inspect
    import sagnac_mcts_planner as sp
    src = inspect.getsource(sp)
    assert "demo_token_ids = [0]" not in src, "all-zero bootstrap labels must not return"
    assert "adapt_in_context_sgld_wave" in src, "corrected soft-target SGLD must be wired"
    assert "self.decoder.adapt_in_context(demo_waves" not in src, "CE-only inert call must not return"


def test_text_egress_hopfield_snap_retrieval():
    """UniversalEgress remembering probe: a query near an exemplar snaps to it."""
    import torch.nn.functional as F
    from henri_egress import TextEgress
    torch.manual_seed(3)
    eg = TextEgress(d_model=256, beta=8.0)
    sols = ["def f():\n    return 1", "def g():\n    return 2", "def h():\n    return 3"]
    base = torch.randn(3, 256)
    base = F.normalize(base, dim=-1)
    eg.register_tokens(base, sols)
    q = base[1] + 0.01 * torch.randn(256)
    q = F.normalize(q, dim=-1)
    text, idx, sim = eg.decode_wave(q)
    assert int(idx) == 1
    assert text == sols[1]
    assert float(sim) > 0.9


def test_recursive_edmd_kill_gate_mechanics():
    """c3-next kill-test mechanics at reduced dim: the online R-EDMD operator
    must (a) beat the identity operator on exemplar self-prediction when a
    transformation is learnable (the pilot's gate), and (b) generalize
    leave-one-out better on a structured map than on random targets."""
    from recursive_dual_edmd import RecursiveDualEDMD
    d, r, n = 128, 16, 8
    torch.manual_seed(11)
    edmd = RecursiveDualEDMD(d_model=d, r_rank=r)
    edmd_id = RecursiveDualEDMD(d_model=d, r_rank=r)
    w0 = torch.nn.functional.normalize(torch.randn(d), dim=0)
    xs = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    a = torch.zeros(d)
    ys = torch.nn.functional.normalize(0.7 * xs + 0.3 * w0, dim=-1)  # structured attractor map
    ys_rand = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    for i in range(n):
        edmd.update_online_step(xs[i], a, ys[i])
    def self_sim(pred, x_list, y_list):
        sims = []
        for x, y in zip(x_list, y_list):
            p = pred(x, a)
            sims.append(float(torch.dot(p, y).item()))
        return sum(sims) / len(sims)
    learned = self_sim(edmd, xs, ys)
    identity = self_sim(edmd_id, xs, ys)
    assert learned > identity + 0.02, f"gate: learned={learned:.4f} identity={identity:.4f}"
    def loo_cv(targets):
        holdout = []
        for k in range(n):
            m = RecursiveDualEDMD(d_model=d, r_rank=r)
            for i in range(n):
                if i != k:
                    m.update_online_step(xs[i], a, targets[i])
            p = m(xs[k], a)
            holdout.append(float(torch.dot(p, targets[k]).item()))
        return sum(holdout) / len(holdout)
    loo_learned = loo_cv(ys)
    loo_rand = loo_cv(ys_rand)
    assert loo_learned > loo_rand + 0.05, f"loo: learned={loo_learned:.4f} random={loo_rand:.4f}"


def test_recursive_edmd_manifold_basis_captures_transition():
    """c3-next run9: seeding V from the exemplar manifold (the EDMD dictionary)
    must let the operator strongly beat the identity operator — the run8
    random-basis underfit is fixed by acting within the observable span."""
    from recursive_dual_edmd import RecursiveDualEDMD
    d, r, n = 128, 16, 8
    torch.manual_seed(13)
    w0 = torch.nn.functional.normalize(torch.randn(d), dim=0)
    xs = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    ys = torch.nn.functional.normalize(0.7 * xs + 0.3 * w0, dim=-1)
    a = torch.zeros(d)
    _, _, Vt = torch.linalg.svd(torch.stack(list(xs) + list(ys)), full_matrices=False)
    v_basis = Vt.T[:, :r].contiguous()
    edmd = RecursiveDualEDMD(d_model=d, r_rank=r, v_basis=v_basis)
    edmd_id = RecursiveDualEDMD(d_model=d, r_rank=r, v_basis=v_basis)
    for i in range(n):
        edmd.update_online_step(xs[i], a, ys[i])
    def self_sim(pred, x_list, y_list):
        sims = []
        for x, y in zip(x_list, y_list):
            p = pred(x, a)
            sims.append(float(torch.dot(p, y).item()))
        return sum(sims) / len(sims)
    learned = self_sim(edmd, xs, ys)
    identity = self_sim(edmd_id, xs, ys)
    assert learned > identity + 0.05, f"manifold: learned={learned:.4f} identity={identity:.4f}"
    assert learned > 0.5, f"manifold: learned={learned:.4f} — targets in-span but not recovered"


def test_cegis_parse_entry_signature():
    from mbpp_cegis_synthesizer import parse_entry_from_tests, parse_entry_signature
    assert parse_entry_signature("def add_binary(a, b):\n    pass\n") == ("add_binary", ["a", "b"])
    assert parse_entry_signature("no def here") is None
    assert parse_entry_signature("def f(x, y, z):\n    return x") == ("f", ["x", "y", "z"])
    # MBPP prompts have no def line: signature comes from the test calls
    assert parse_entry_from_tests(["assert small_nnum([5, 6], 1) == [5]"]) == ("small_nnum", ["a0", "a1"])
    assert parse_entry_from_tests(["assert is_not_prime(2) == False"]) == ("is_not_prime", ["a0"])


def test_cegis_build_candidates_renames_args_and_wraps():
    """AST-level arg/entry renaming + wrapper morphisms, all syntax-valid."""
    from mbpp_cegis_synthesizer import MbppCegisSynthesizer
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
    ex = [{"task_id": 1, "code": "def small_nnum(list1, n):\n    return list1[:n]"}]
    codec = qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")
    synth = MbppCegisSynthesizer(ex, codec, device="cpu")
    # no def line in the prompt: signature must come from the test list
    cands = synth.build_candidates("Write a function to get the k smallest.\n", ["assert k_smallest([5, 6, 1], 2) == [1, 5]"])
    assert len(cands) == 5  # identity + 4 wrappers
    sources = [c[0] for c in cands]
    assert "def k_smallest(a0, a1):" in sources[0]
    assert "a0[:a1]" in sources[0]
    assert "return list(a0[:a1])" in sources[1]
    assert "return sorted(a0[:a1])" in sources[3]
    for s in sources:
        import ast
        ast.parse(s)  # every candidate is syntax-valid


def test_cegis_ranking_prefers_true_solution():
    """The predicted wave ranks the true solution first among distractors."""
    from mbpp_cegis_synthesizer import MbppCegisSynthesizer
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
    import torch
    codec = qFHRREpistemicCodec(d_model=2048, k_bins=256, device="cpu")
    exs = [
        {"task_id": 1, "code": "def f1(x):\n    return [i * 2 for i in x]"},
        {"task_id": 2, "code": "def f2(x):\n    return sum(x)"},
        {"task_id": 3, "code": "def f3(x):\n    return sorted(x, reverse=True)"},
    ]
    synth = MbppCegisSynthesizer(exs, codec, device="cpu")
    prompt = "def solve(x):\n    pass\n"
    cands = synth.build_candidates(prompt)
    true_src = "def solve(x):\n    return sorted(x, reverse=True)"
    ring = codec.encode_text(true_src).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
    pred = torch.nn.functional.normalize(ring.view(-1), p=2, dim=0)
    ranked = synth.rank_candidates(cands, pred)
    assert ranked[0][0] == true_src, f"top candidate {ranked[0][0]!r} != true {true_src!r}"


def test_cegis_verify_loop_orders_by_tests():
    """The CEGIS loop returns the first candidate that passes ALL tests."""
    from mbpp_cegis_synthesizer import MbppCegisSynthesizer
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

    class _StubSandbox:
        def __init__(self, passing_src: str):
            self.passing_src = passing_src

        def execute(self, code: str):
            class _R:
                status = "PASS" if self.passing_src in code else "FAIL"
            return _R()

    codec = qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")
    exs = [{"task_id": 1, "code": "def f(x):\n    return x"}]
    synth = MbppCegisSynthesizer(exs, codec, device="cpu")
    winner = "def solve(x):\n    return [v for v in x if v > 0]"
    ranked = [
        ("def solve(x):\n    return x", {}, 0.9),
        (winner, {}, 0.8),
        ("def solve(x):\n    return []", {}, 0.7),
    ]
    sb = _StubSandbox(winner)
    code, meta = synth.cegis_verify(ranked, {"test_list": ["check(True)"]}, sb, max_attempts=3)
    assert code == winner
    assert meta["candidates_tried"] == 2
    assert meta["cegis"] is True


def test_cegis_verify_returns_none_when_nothing_passes():
    from mbpp_cegis_synthesizer import MbppCegisSynthesizer
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

    class _StubSandbox:
        def execute(self, code):
            class _R:
                status = "FAIL"
            return _R()

    codec = qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")
    synth = MbppCegisSynthesizer([{"task_id": 1, "code": "def f(x):\n    return x"}], codec, device="cpu")
    ranked = [("def solve(x):\n    return x", {}, 0.5)]
    code, meta = synth.cegis_verify(ranked, {"test_list": []}, _StubSandbox(), max_attempts=1)
    assert code is None
    assert meta["cegis"] is False


def test_cegis_transformation_ranking_resists_manifold_blend():
    """run11 diagnosis: the R-EDMD prediction is a holographic blend in the
    exemplar manifold. With prompt-relative (transformation) ranking, the true
    solution must win even when the predicted wave blends it with a distractor."""
    from mbpp_cegis_synthesizer import MbppCegisSynthesizer
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
    import torch
    codec = qFHRREpistemicCodec(d_model=2048, k_bins=256, device="cpu")
    exs = [
        {"task_id": 1, "code": "def f1(x):\n    return [i * 2 for i in x]"},
        {"task_id": 2, "code": "def f2(x):\n    return sum(x)"},
        {"task_id": 3, "code": "def f3(x):\n    return sorted(x, reverse=True)"},
    ]
    synth = MbppCegisSynthesizer(exs, codec, device="cpu")
    prompt = "def solve(x):\n    pass\n"
    cands = synth.build_candidates(prompt)
    prompt_wave = codec.encode_text(prompt).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
    true_src = "def solve(x):\n    return sorted(x, reverse=True)"
    distractor_src = "def solve(x):\n    return sum(x)"
    def _real(src):
        return codec.encode_text(src).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
    # blend: 0.8 true + 0.2 distractor (holographic superposition)
    pred_blend = 0.8 * torch.nn.functional.normalize(_real(true_src).view(-1), p=2, dim=0) + \
                 0.2 * torch.nn.functional.normalize(_real(distractor_src).view(-1), p=2, dim=0)
    ranked = synth.rank_candidates(cands, pred_blend, prompt_wave=prompt_wave.view(-1))
    assert ranked[0][0] == true_src, f"blend top candidate {ranked[0][0]!r} != true"


def test_w_task_retrieval_captures_consistent_rule():
    codec = qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")
    comp = HolographicTaskFunctorCompiler(codec)
    demos = _rule_demo_pairs(n=8, d_model=1024, shift=3)
    w_task = comp.compile_functor(demos)
    goal = comp.single_pass_associative_retrieval(w_task, demos[0][0])
    sim = _ring_cosine_sim(goal, demos[0][1])
    assert sim > 0.5, f"consistent rule should retrieve its own output (sim={sim:.3f})"
    # generalization: retrieve the 9th unseen input, compare to its rule output
    x_new = torch.randint(0, 256, (1024,), dtype=torch.uint8)
    y_new = ((x_new.to(torch.int32) + 3) % 256).to(torch.uint8)
    goal_new = comp.single_pass_associative_retrieval(w_task, x_new)
    sim_new = _ring_cosine_sim(goal_new, y_new)
    assert sim_new > 0.5, f"rule should generalize (sim={sim_new:.3f})"


def test_w_task_no_rule_falls_below_kill_gate():
    codec = qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")
    comp = HolographicTaskFunctorCompiler(codec)
    demos = _random_demo_pairs(n=8, d_model=1024)
    w_task = comp.compile_functor(demos)
    goal = comp.single_pass_associative_retrieval(w_task, demos[0][0])
    sim = _ring_cosine_sim(goal, demos[0][1])
    assert sim < 0.5, f"no-rule control must fail the kill gate (sim={sim:.3f})"


def test_w_task_gate_logic_matches_runner():
    """Mirror of the production gate: adopt only when sim >= threshold."""
    codec = qFHRREpistemicCodec(d_model=1024, k_bins=256, device="cpu")
    comp = HolographicTaskFunctorCompiler(codec)
    good = comp.compile_functor(_rule_demo_pairs(n=8, d_model=1024, shift=3))
    bad = comp.compile_functor(_random_demo_pairs(n=8, d_model=1024))
    g_good = comp.single_pass_associative_retrieval(good, _rule_demo_pairs(1, 1024, 3)[0][0])
    g_bad = comp.single_pass_associative_retrieval(bad, _random_demo_pairs(1, 1024)[0][0])
    sim_good = _ring_cosine_sim(g_good, _rule_demo_pairs(1, 1024, 3)[0][1])
    sim_bad = _ring_cosine_sim(g_bad, _random_demo_pairs(1, 1024)[0][1])
    assert (sim_good >= 0.5) and (sim_bad < 0.5)
