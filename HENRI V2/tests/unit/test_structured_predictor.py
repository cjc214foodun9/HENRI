"""Contract tests for the structured (tripartite) resonator predictor.

These pin the properties the module docstring CLAIMS, so the code cannot drift
into claiming more than it does. The two honesty-relevant tests are
`test_rotor_is_exactly_orthogonal` and `test_no_claim_of_exact_norm_preservation`
-- if the factorisation stopped being orthogonal, or if the output stopped being
norm-restored at the boundary, these fail rather than silently degrading.

Evidence class: these are CPU unit tests. They establish INTERNAL properties of
the module. They are NOT a task outcome and NOT a 0.92-gate claim.
"""

import math

import torch
import torch.nn.functional as F

import henri_structured_predictor as SP


NB, SLOTS = 64, 8
D = NB * SLOTS


def test_forward_shape_2d_and_3d():
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, iters=3)
    x2 = torch.randn(D)
    x3 = torch.randn(5, D)
    assert m(x2).shape == (D,)
    assert m(x3).shape == (5, D)


def test_output_is_unit_norm_at_the_boundary():
    """The module claims norm-RESTORATION at the boundary (not preservation)."""
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, iters=3)
    x = torch.randn(4, D)
    y = m(x)
    n = y.norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5), n.tolist()


def test_rotor_is_exactly_orthogonal():
    """R = expm(S - S^T) must be orthogonal; x @ R exactly norm-preserving."""
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16)
    torch.nn.init.normal_(m.rotor.S, std=1.0)      # non-trivial generator
    R = m.rotor.matrix()
    err = (R @ R.t() - torch.eye(SLOTS)).abs().max()
    assert float(err) < 1e-5, float(err)
    h = torch.randn(3, NB, SLOTS)
    d = (h.norm(dim=-1) - (h @ R).norm(dim=-1)).abs().max()
    assert float(d) < 1e-5, float(d)


def test_block_shift_is_an_exact_permutation():
    """T must preserve the multiset of block vectors (a permutation, not a mix)."""
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16)
    h = torch.randn(NB, SLOTS)
    t = m.shift(h.unsqueeze(0)).squeeze(0)
    a = h.norm(dim=-1).sort().values
    b = t.norm(dim=-1).sort().values
    assert torch.allclose(a, b, atol=1e-6)


def test_all_three_factors_receive_gradient():
    """A factor with zero gradient is decorative -- assert each one is engaged."""
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, iters=3)
    m(torch.randn(2, D)).sum().backward()
    for name in ("rotor.S", "mask_logits", "alpha", "shift.logits"):
        p = dict(m.named_parameters())[name]
        g = 0.0 if p.grad is None else float(p.grad.abs().sum())
        assert g > 0.0, f"{name} received no gradient (decorative factor)"


def test_non_square_block_count_is_rejected():
    """The spatial factor needs a square block ring; refuse rather than guess."""
    try:
        SP.StructuredResonator(d_model=60, num_blocks=60, grid=16)
        raise AssertionError("non-square num_blocks must raise")
    except ValueError:
        pass


def test_no_claim_of_exact_norm_preservation():
    """Guard against a future edit that removes the boundary renormalisation.

    If `normalize` is dropped the output norm is no longer 1, which would make
    the module's stated contract false. This test detects that drift.
    """
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, iters=3)
    x = torch.randn(1, D)
    with torch.no_grad():
        m.alpha.fill_(25.0)            # amplify: pre-normalise norm diverges
    y = m(x)
    assert abs(float(y.norm()) - 1.0) < 1e-5


def test_build_arm_factory_contract():
    m = SP.build_arm(D, NB, grid=16, iters=2)
    assert m.iters == 2
    assert m.factors() == ("R_rotor", "Pi_mask", "T_blockshift")
