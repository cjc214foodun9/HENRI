"""Phase 8.19 contract tests — SU(3) MCTS planner (C2/C3/C4).

Gates (brief section 3.1):
  G1-8.19 (router): covered by remote runner smoke (env exposes no demos ->
                     SU3_MCTS_ROUTER event), not unit-testable here.
  G2-8.19 (tree):   8-directional Lie generator expansions unitary
                     error < 1e-6 (self-test asserts the same bound).
  C2/C3/C4:         branch shapes, Sagnac veto firing, Langevin branch
                     selection, planner-domain goal wave shape [N, 8] real.
"""
import pytest
import torch

from su3_mcts_planner import SU3MCTSPlanner

try:
    from chromodynamic_grounding import GELL_MANN_BASIS

    _HAS_BASIS = True
except Exception:  # pragma: no cover
    _HAS_BASIS = False


def _unit_field(n: int = 64) -> torch.Tensor:
    """Deterministic det-1 SU(3) field (identity, det-normalized)."""
    torch.manual_seed(5)
    u = torch.eye(3, dtype=torch.complex64).expand(n, 3, 3).clone()
    u = u / torch.linalg.det(u).abs().sqrt().unsqueeze(-1).unsqueeze(-1)
    return u


@pytest.mark.skipif(not _HAS_BASIS, reason="chromodynamic grounding unavailable")
def test_g2_branch_rotations_unitary() -> None:
    planner = SU3MCTSPlanner(GELL_MANN_BASIS.to(torch.complex64), num_channels=64)
    rot = planner.rotations
    unit_err = float(
        (rot.conj().transpose(-1, -2) @ rot
         - torch.eye(3, dtype=rot.dtype)).abs().max().item()
    )
    assert unit_err < 1e-6, f"rotation unitarity err {unit_err} >= 1e-6"


@pytest.mark.skipif(not _HAS_BASIS, reason="chromodynamic grounding unavailable")
def test_expand_counterfactual_branches_shapes() -> None:
    planner = SU3MCTSPlanner(GELL_MANN_BASIS.to(torch.complex64), num_channels=64)
    u = _unit_field(64)
    out = planner.expand_counterfactual_branches(u)
    assert tuple(out.shape) == (8, 64, 3, 3)
    batched = u.unsqueeze(0)  # [1, 64, 3, 3]
    out_b = planner.expand_counterfactual_branches(batched)
    assert tuple(out_b.shape) == (1, 8, 64, 3, 3)


@pytest.mark.skipif(not _HAS_BASIS, reason="chromodynamic grounding unavailable")
def test_sagnac_veto_fires_for_distant_branch() -> None:
    planner = SU3MCTSPlanner(GELL_MANN_BASIS.to(torch.complex64), num_channels=64)
    u = _unit_field(64)
    anchor = planner._field_to_real_wave(u)
    # genuine generator rotation (global U(1) phases cancel in the su(3) log)
    rot_far = torch.matrix_exp(
        (1j * torch.tensor(2.0) * GELL_MANN_BASIS[0]).to(torch.complex64)
    )
    far = rot_far @ u
    far = far / torch.linalg.det(far).abs().sqrt().unsqueeze(-1).unsqueeze(-1)
    d_far, _ = planner.evaluate_node(far, anchor)
    assert d_far > planner.veto_threshold
    d_id, _ = planner.evaluate_node(u, anchor)
    assert d_id <= planner.veto_threshold


@pytest.mark.skipif(not _HAS_BASIS, reason="chromodynamic grounding unavailable")
def test_langevin_selection_prunes_vetoed_branches() -> None:
    planner = SU3MCTSPlanner(GELL_MANN_BASIS.to(torch.complex64), num_channels=64)
    deltas = torch.tensor([0.02, 0.05, 0.5, 0.8, 0.02, 0.04, 0.9, 0.7])
    keep = deltas <= planner.veto_threshold  # branches 0,1,4,5
    chosen = planner._select_branch(deltas, keep)
    assert float(deltas[chosen]) <= planner.veto_threshold
    # all-vetoed: temperature backpropagates, selection still returns an index
    keep_none = torch.zeros_like(keep, dtype=torch.bool)
    chosen_fallback = planner._select_branch(deltas, keep_none)
    assert 0 <= chosen_fallback < 8
    assert planner.temperature > 1.0


@pytest.mark.skipif(not _HAS_BASIS, reason="chromodynamic grounding unavailable")
def test_search_goal_attractor_returns_planner_domain_wave() -> None:
    planner = SU3MCTSPlanner(GELL_MANN_BASIS.to(torch.complex64), num_channels=64)
    u = _unit_field(64)
    w_goal = planner.search_goal_attractor(u, max_rollouts=8)
    assert tuple(w_goal.shape) == (64, 8)
    assert w_goal.is_floating_point()
    assert bool(torch.isfinite(w_goal).all())
