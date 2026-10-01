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


# ---------------------------------------------------------------------------
# THE ALGEBRAIC GUARD. This is the test the original module lacked: it pins why
# the first `resonator` arm landed BELOW the linear reference (0.088508 vs
# 0.145739) rather than merely short of 0.92.
#
# A linear map L satisfies dir(L(c*u)) == dir(L(u)) for every c > 0, exactly,
# after normalisation. Cosine similarity reads DIRECTION only, so any composite
# that is linear in direction has its expressiveness capped by the linear
# function class -- the 0.280/0.232 model-class ceiling.
#
# Parametrised so BOTH facts are asserted: the no-mixer arm IS linear in
# direction (the defect, documented), and the mixer arm is NOT.
# ---------------------------------------------------------------------------

def _dir_deviation(m, c: float = 3.0) -> float:
    """|| dir(m(u)) - dir(m(c*u)) ||, the direction-proportionality residual."""
    torch.manual_seed(7)
    u = torch.randn(D)
    a = F.normalize(m(u), dim=-1)
    b = F.normalize(m(c * u), dim=-1)
    return float((a - b).norm())


def test_composite_breaks_direction_proportionality():
    # (1) no-mixer arm: exactly linear in direction -> residual ~ 0
    lin = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, iters=3,
                                 slot_mixer=False)
    torch.nn.init.normal_(lin.rotor.S, std=1.0)
    assert _dir_deviation(lin) < 1e-5, _dir_deviation(lin)

    # (2) mixer arm: must leave the linear class. NOTE the mixer output layer is
    # ZERO-initialised so the arm STARTS at the linear solution (a deliberate A/B
    # choice), which makes the residual ~0 AT INIT by construction. Perturb it
    # off zero-init before asserting, otherwise this test measures the
    # initialisation, not the architecture -- measured trap: the first revision
    # of this guard reported 8.89e-08 for BOTH arms for exactly that reason.
    nl = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, iters=3,
                                slot_mixer=True, mixer_hidden=32)
    torch.nn.init.normal_(nl.mixer.net[-1].weight, std=0.5)
    torch.nn.init.normal_(nl.mixer.net[-1].bias, std=0.1)
    assert _dir_deviation(nl) > 1e-3, _dir_deviation(nl)


def test_mixer_output_is_zero_initialised():
    """The A/B depends on the mixer arm STARTING at the linear solution.

    BUG IN AN EARLIER REVISION OF THIS TEST: it built `nl` and `lin` separately
    and compared their outputs directly. Each construction draws a DIFFERENT
    random rotor/mask, so the comparison was false for an unrelated reason and
    the test failed on a correct implementation. The fix is to copy the linear
    arm's parameters into the mixer arm before comparing, which is what the
    property actually means: same parameters, mixer contributes zero.
    """
    lin = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, slot_mixer=False)
    nl = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, slot_mixer=True)
    missing, unexpected = nl.load_state_dict(lin.state_dict(), strict=False)
    assert not unexpected, unexpected          # nothing left over from `lin`
    assert all(k.startswith("mixer.") for k in missing), missing
    last = nl.mixer.net[-1]
    assert float(last.weight.abs().max()) == 0.0
    assert float(last.bias.abs().max()) == 0.0
    torch.manual_seed(3)
    u = torch.randn(D)
    assert torch.allclose(nl(u), lin(u), atol=1e-6), "arm must start at the linear solution"


def test_mixer_engages_after_one_optimiser_step():
    """Zero-init starves the FIRST layer at step 0. Assert it recovers.

    With the output layer zero-initialised, mixer(x) == 0 at init, so
    mixer.net[0].weight receives ZERO gradient on the first step. That is the
    intended ReZero-style start, but it would be a silent defect if the layer
    stayed dead. This test pins that it comes alive after one update.
    """
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16, iters=1, slot_mixer=True)
    x = torch.randn(4, D)
    y = torch.randn(4, D)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(3):
        loss = F.mse_loss(F.normalize(m(x), dim=-1), F.normalize(y, dim=-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert m.mixer.net[0].weight.grad is not None
    assert float(m.mixer.net[0].weight.grad.abs().sum()) > 0.0, "first layer is dead"


def test_mixer_param_count_is_independent_of_num_blocks():
    """At production nb = 8192 a per-block module would reintroduce the memory
    wall the factorisation exists to avoid. The mixer is SHARED across blocks."""
    small = SP.StructuredResonator(d_model=64 * 8, num_blocks=64, grid=16, slot_mixer=True)
    big = SP.StructuredResonator(d_model=256 * 8, num_blocks=256, grid=16, slot_mixer=True)
    ms = sum(p.numel() for p in small.mixer.parameters())
    mb = sum(p.numel() for p in big.mixer.parameters())
    assert ms == mb, (ms, mb)
    assert ms > 0


def test_slot_mixer_is_default_off():
    """Default-off keeps the committed `resonator` arm byte-reproducible."""
    m = SP.StructuredResonator(d_model=D, num_blocks=NB, grid=16)
    assert m.use_slot_mixer is False
    assert m.mixer is None
    assert "M_slotmixer" not in m.factors()
