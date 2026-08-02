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
