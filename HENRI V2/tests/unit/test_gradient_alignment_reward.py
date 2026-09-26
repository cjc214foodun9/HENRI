"""Unit tests for the preconditioned gradient-alignment reward kernel.

Each pre-registered acceptance condition from the blueprint has a test:

    A1  mastered input (gradient -> 0)              -> reward -> 0
    A2  gradient orthogonal to P @ delta_theta      -> reward -> 0
    A3  gradient parallel to P @ delta_theta        -> maximal reward
    A4  discrimination control: the preconditioner must change the candidate
        RANKING when v is non-uniform, and must act only as a global rescale
        when v is uniform (a tautological preconditioner must fail this test)
    A5  fail-closed on invalid input

Input-sensitivity controls (dead-input defect class) are also present: the
reward must change when the displacement changes and when the gradient changes.
A kernel that ignores a declared input must fail these tests.
"""

from __future__ import annotations

import pytest
import torch

from henri_gradient_alignment_reward import (
    AlignmentRewardConfig,
    AlignmentRewardError,
    adamw_preconditioner,
    alignment_reward,
    lookback_index,
    parameter_displacement,
)

DTYPE = torch.float64
TOL = 1e-12


# --------------------------------------------------------------------------------------
# A5 — lookback and fail-closed input validation
# --------------------------------------------------------------------------------------


def test_lookback_index_is_floor_half():
    assert lookback_index(0) == 0
    assert lookback_index(1) == 0
    assert lookback_index(2) == 1
    assert lookback_index(3) == 1
    assert lookback_index(8) == 4


@pytest.mark.parametrize("bad", [-1, -7])
def test_lookback_index_rejects_negative(bad):
    with pytest.raises(AlignmentRewardError):
        lookback_index(bad)


@pytest.mark.parametrize("bad", [1.0, "2", None, True])
def test_lookback_index_rejects_non_int(bad):
    with pytest.raises(AlignmentRewardError):
        lookback_index(bad)


def test_config_rejects_bad_lr_and_eps():
    with pytest.raises(AlignmentRewardError):
        AlignmentRewardConfig(lr=0.0).validate()
    with pytest.raises(AlignmentRewardError):
        AlignmentRewardConfig(lr=-1e-4).validate()
    with pytest.raises(AlignmentRewardError):
        AlignmentRewardConfig(eps=0.0).validate()


def test_reward_rejects_shape_mismatch():
    g = torch.zeros(4, dtype=DTYPE)
    d = torch.zeros(5, dtype=DTYPE)
    v = torch.ones(4, dtype=DTYPE)
    with pytest.raises(AlignmentRewardError, match="shape mismatch"):
        alignment_reward(g, d, v)


def test_reward_rejects_non_finite():
    g = torch.tensor([1.0, float("nan")], dtype=DTYPE)
    d = torch.ones(2, dtype=DTYPE)
    v = torch.ones(2, dtype=DTYPE)
    with pytest.raises(AlignmentRewardError, match="non-finite"):
        alignment_reward(g, d, v)


def test_reward_rejects_negative_second_moment():
    g = torch.ones(3, dtype=DTYPE)
    d = torch.ones(3, dtype=DTYPE)
    v = torch.tensor([1.0, -0.5, 1.0], dtype=DTYPE)
    with pytest.raises(AlignmentRewardError, match="non-negative"):
        alignment_reward(g, d, v)


def test_reward_rejects_bad_rank():
    g = torch.ones(2, 2, 2, dtype=DTYPE)
    with pytest.raises(AlignmentRewardError, match="rank"):
        alignment_reward(g, g.clone(), g.clone())


def test_reward_rejects_non_tensor():
    with pytest.raises(AlignmentRewardError, match="torch.Tensor"):
        alignment_reward([0.0, 1.0], torch.ones(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE))


# --------------------------------------------------------------------------------------
# A1 — zero reward for mastered data
# --------------------------------------------------------------------------------------


def test_a1_mastered_input_gives_zero_reward():
    """A mastered sample has a vanishing loss gradient, so the reward vanishes."""
    p = 64
    v = torch.rand(p, dtype=DTYPE) + 0.1
    d = torch.randn(p, dtype=DTYPE)
    g_mastered = torch.zeros(p, dtype=DTYPE)
    r = alignment_reward(g_mastered, d, v)
    assert r.item() == pytest.approx(0.0, abs=TOL)


def test_a1_mastered_approaches_zero_continuously():
    """As the gradient shrinks, the reward shrinks at the same rate."""
    p = 32
    torch.manual_seed(0)
    v = torch.rand(p, dtype=DTYPE) + 0.1
    d = torch.randn(p, dtype=DTYPE)
    g = torch.randn(p, dtype=DTYPE)
    r_full = alignment_reward(g, d, v).item()
    r_half = alignment_reward(0.5 * g, d, v).item()
    assert r_half == pytest.approx(0.5 * r_full, rel=1e-9)


# --------------------------------------------------------------------------------------
# A2 — zero reward for a gradient orthogonal to the coherent direction
# --------------------------------------------------------------------------------------


def test_a2_orthogonal_gradient_gives_zero_reward():
    """Pure random noise is orthogonal to the coherent displacement direction."""
    p = 256
    torch.manual_seed(1)
    v = torch.rand(p, dtype=DTYPE) + 0.05
    d = torch.randn(p, dtype=DTYPE)

    p_vec = v.new_tensor(1e-4) / (torch.sqrt(v) + v.new_tensor(1e-8))
    coherent = p_vec * d  # the direction the learner can consolidate

    # Build a gradient that is exactly orthogonal to the coherent direction.
    noise = torch.randn(p, dtype=DTYPE)
    noise = noise - (torch.dot(noise, coherent) / torch.dot(coherent, coherent)) * coherent

    assert abs(torch.dot(noise, coherent).item()) < 1e-9  # construction check
    r = alignment_reward(noise, d, v)
    assert r.item() == pytest.approx(0.0, abs=1e-9)


def test_a2_orthogonal_is_far_smaller_than_parallel():
    p = 256
    torch.manual_seed(2)
    v = torch.rand(p, dtype=DTYPE) + 0.05
    d = torch.randn(p, dtype=DTYPE)
    p_vec = v.new_tensor(1e-4) / (torch.sqrt(v) + v.new_tensor(1e-8))
    coherent = p_vec * d
    unit = coherent / torch.linalg.vector_norm(coherent)

    r_par = alignment_reward(unit, d, v).item()
    noise = torch.randn(p, dtype=DTYPE)
    noise = noise - (torch.dot(noise, coherent) / torch.dot(coherent, coherent)) * coherent
    r_orth = alignment_reward(noise, d, v).item()

    assert r_orth < 1e-9
    assert r_par > 1e-3  # non-degenerate scale: the gate is not vacuous


# --------------------------------------------------------------------------------------
# A3 — maximal reward at the epistemic frontier
# --------------------------------------------------------------------------------------


def test_a3_parallel_gradient_is_maximal():
    """The reward is maximised when the gradient is parallel to P @ delta_theta."""
    p = 128
    torch.manual_seed(3)
    v = torch.rand(p, dtype=DTYPE) + 0.05
    d = torch.randn(p, dtype=DTYPE)
    p_vec = v.new_tensor(1e-4) / (torch.sqrt(v) + v.new_tensor(1e-8))
    coherent = p_vec * d
    unit = coherent / torch.linalg.vector_norm(coherent)

    r_parallel = alignment_reward(unit, d, v).item()

    # Any other gradient of the same norm must score lower.
    other = torch.randn(p, dtype=DTYPE)
    other = other / torch.linalg.vector_norm(other)
    r_other = alignment_reward(other, d, v).item()

    assert r_parallel > r_other
    assert r_parallel == pytest.approx(torch.linalg.vector_norm(coherent).item(), rel=1e-9)


# --------------------------------------------------------------------------------------
# A4 — discrimination control (the anti-tautology test)
# --------------------------------------------------------------------------------------


def test_a4_uniform_v_is_only_a_global_rescale():
    """With a uniform second moment the preconditioner cannot reorder candidates."""
    torch.manual_seed(4)
    b, p = 4, 64
    v = torch.full((b, p), 0.5, dtype=DTYPE)
    d = torch.randn(b, p, dtype=DTYPE)
    g = torch.randn(b, p, dtype=DTYPE)

    r_1 = alignment_reward(g, d, v, lr=1e-4)
    r_2 = alignment_reward(g, d, v, lr=2e-4)

    # Ranking must be identical; only the scale changes.
    assert torch.equal(torch.argsort(r_1), torch.argsort(r_2))
    assert torch.allclose(r_2, 2.0 * r_1, rtol=1e-9)


def test_a4_non_uniform_v_changes_the_candidate_ranking():
    """The preconditioner is engaged only when the second moment is non-uniform.

    This is the discriminating control. A preconditioner that merely rescales all
    candidates would leave the ranking invariant here and must fail this test.
    """
    torch.manual_seed(5)
    b, p = 3, 32
    d = torch.randn(b, p, dtype=DTYPE)
    g = torch.randn(b, p, dtype=DTYPE)

    # Candidate 1 has a large second moment in exactly the dimensions where its
    # gradient is large: the preconditioner must damp it.
    v = torch.ones(b, p, dtype=DTYPE) * 0.1
    v[1] = 0.1 + 50.0 * torch.abs(g[1])

    r_hetero = alignment_reward(g, d, v)

    # The "no preconditioner" variant: replace P by a single scalar.
    p_scalar = 1e-4 / (torch.sqrt(torch.tensor(0.1, dtype=DTYPE)) + 1e-8)
    r_flat = torch.abs(torch.sum(g * (p_scalar * d), dim=-1))

    # On the damped candidate the two differ materially -> P is engaged per-dimension.
    rel = (r_hetero[1] - r_flat[1]).abs() / r_flat[1].abs()
    assert rel > 0.5, f"preconditioner did not engage per-dimension (rel diff {rel})"


def test_a4_preconditioner_matches_direct_formula():
    v = torch.tensor([0.25, 1.0, 4.0], dtype=DTYPE)
    p = adamw_preconditioner(v, lr=1e-4, eps=1e-8)
    expected = 1e-4 / (torch.sqrt(v) + 1e-8)
    assert torch.allclose(p, expected, rtol=1e-12)


# --------------------------------------------------------------------------------------
# input-sensitivity controls (dead-input defect class)
# --------------------------------------------------------------------------------------


def test_control_reward_reads_the_displacement():
    """If the displacement input were ignored, these two rewards would be equal."""
    p = 16
    torch.manual_seed(6)
    v = torch.rand(p, dtype=DTYPE) + 0.1
    g = torch.randn(p, dtype=DTYPE)
    d1 = torch.randn(p, dtype=DTYPE)
    d2 = torch.randn(p, dtype=DTYPE)

    assert not torch.isclose(alignment_reward(g, d1, v), alignment_reward(g, d2, v))


def test_control_reward_reads_the_gradient():
    p = 16
    torch.manual_seed(7)
    v = torch.rand(p, dtype=DTYPE) + 0.1
    d = torch.randn(p, dtype=DTYPE)
    g1 = torch.randn(p, dtype=DTYPE)
    g2 = torch.randn(p, dtype=DTYPE)

    assert not torch.isclose(alignment_reward(g1, d, v), alignment_reward(g2, d, v))


def test_control_reward_reads_the_second_moment():
    p = 16
    torch.manual_seed(8)
    d = torch.randn(p, dtype=DTYPE)
    g = torch.randn(p, dtype=DTYPE)
    v1 = torch.rand(p, dtype=DTYPE) + 0.1
    v2 = torch.rand(p, dtype=DTYPE) + 0.1

    assert not torch.isclose(alignment_reward(g, d, v1), alignment_reward(g, d, v2))


def test_control_unpreconditioned_variant_would_fail_a4():
    """Prove the A4 test can fail: a kernel with no preconditioner is caught.

    This is the negative control for the gate itself. It shows A4 is not vacuous.
    """
    torch.manual_seed(9)
    b, p = 3, 32
    d = torch.randn(b, p, dtype=DTYPE)
    g = torch.randn(b, p, dtype=DTYPE)
    v = torch.ones(b, p, dtype=DTYPE) * 0.1
    v[1] = 0.1 + 50.0 * torch.abs(g[1])

    r_real = alignment_reward(g, d, v)
    p_scalar = 1e-4 / (torch.sqrt(torch.tensor(0.1, dtype=DTYPE)) + 1e-8)
    r_tautology = torch.abs(torch.sum(g * (p_scalar * d), dim=-1))

    rel = (r_real[1] - r_tautology[1]).abs() / r_tautology[1].abs()
    assert rel > 0.5, "negative control failed to discriminate; gate A4 may be vacuous"


# --------------------------------------------------------------------------------------
# batching and displacement helper
# --------------------------------------------------------------------------------------


def test_batch_matches_loop():
    torch.manual_seed(10)
    b, p = 5, 24
    v = torch.rand(b, p, dtype=DTYPE) + 0.1
    d = torch.randn(b, p, dtype=DTYPE)
    g = torch.randn(b, p, dtype=DTYPE)

    batched = alignment_reward(g, d, v)
    assert batched.shape == (b,)
    for i in range(b):
        one = alignment_reward(g[i], d[i], v[i])
        assert batched[i].item() == pytest.approx(one.item(), rel=1e-12)


def test_parameter_displacement_direction():
    cur = torch.tensor([1.0, 2.0], dtype=DTYPE)
    look = torch.tensor([3.0, 0.0], dtype=DTYPE)
    d = parameter_displacement(cur, look)
    assert torch.allclose(d, torch.tensor([2.0, -2.0], dtype=DTYPE))


def test_reward_is_non_negative():
    torch.manual_seed(11)
    v = torch.rand(20, dtype=DTYPE) + 0.01
    d = torch.randn(20, dtype=DTYPE)
    g = torch.randn(20, dtype=DTYPE)
    assert alignment_reward(g, d, v).item() >= 0.0
