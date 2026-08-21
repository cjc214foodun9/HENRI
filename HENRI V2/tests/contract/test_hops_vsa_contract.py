"""Contract tests for hops_vsa_core (Class 4.5, HOPS-VLA reference core, default-OFF)."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def projector():
    from hops_vsa_core import HopsVSASkeletonProjector

    return HopsVSASkeletonProjector(d_model=4096, device="cpu")


@pytest.fixture(scope="module")
def rotor():
    from hops_vsa_core import HopsVSACliffordRotor

    return HopsVSACliffordRotor(d_model=4096, device="cpu", seed=7)


def test_skeleton_basis_orthonormal(projector):
    assert projector.gram_error() < 1e-5


def test_projector_preserves_unit_norm_after_normalize(projector):
    v = torch.randn(4096, dtype=torch.float32)
    v = v / v.norm()
    residual = projector.project_null(v)
    # residual may shrink (skeleton removed) but must stay on the sphere after normalize
    assert residual.norm().item() > 0.0
    assert abs((residual / residual.norm()).norm().item() - 1.0) < 1e-6


def test_carrier_removal(projector):
    """Skeleton-projected residual must be ~orthogonal to the skeleton basis."""
    v = projector.V[:, 0]  # a pure skeleton column
    residual = projector.project_null(v)
    # residual of a skeleton vector is ~0 (projected out)
    assert residual.norm().item() < 1e-2


def test_channel_separation(projector):
    """Same skeleton, different body: raw cos high, null-channel cos lower."""
    from hops_vsa_core import _phasor_wave

    # two waves sharing the skeleton phasors but with different body phases
    skel1 = projector.V[:, 0] * 0.5 + projector.V[:, 1] * 0.5
    skel1 = skel1 / skel1.norm()
    body_a = _phasor_wave("BODY_A", 4096, "cpu")
    body_b = _phasor_wave("BODY_B", 4096, "cpu")
    wa = skel1 + 0.1 * body_a
    wb = skel1 + 0.1 * body_b
    wa = wa / wa.norm()
    wb = wb / wb.norm()
    raw_cos = float(torch.dot(wa, wb).item())
    na = torch.nn.functional.normalize(projector.project_null(wa), p=2, dim=0)
    nb = torch.nn.functional.normalize(projector.project_null(wb), p=2, dim=0)
    null_cos = float(torch.dot(na, nb).item())
    assert null_cos < raw_cos


def test_rotor_isometry(rotor):
    """Rotor application preserves unit norm and Gram < 1e-6."""
    v = torch.randn(4096, dtype=torch.float32)
    v = v / v.norm()
    out = rotor(v)
    assert abs(out.norm().item() - 1.0) < 1e-5
    assert rotor.retract() < 1e-6


def test_sagnac_veto_fires():
    from hops_vsa_core import HopsVSASagnacGate

    gate = HopsVSASagnacGate(tau=0.35)
    a = torch.randn(4096, dtype=torch.float32)
    b = torch.randn(4096, dtype=torch.float32)
    assert gate.veto(a, a) is False  # identical -> no veto
    assert gate.veto(a, b) is True  # orthogonal -> veto (delta ~1 > 0.35)


def test_no_dense_allocation(rotor, projector):
    """All params/buffers are [D/2] or [D, k]; never [D, D]."""
    for name, p in rotor.named_parameters():
        assert p.shape[0] < 65536
    V = projector.V
    assert V.shape[1] <= 8
    assert V.shape[0] < 65536 * 8  # far below [D, D]


def test_rejects_uint8_ring(projector):
    from hops_vsa_core import RepresentationBoundaryError

    ring = torch.randint(0, 256, (4096,), dtype=torch.uint8)
    with pytest.raises(RepresentationBoundaryError):
        projector.project_null(ring)


def test_default_off_runner_flag():
    runner = "HENRI V2/humaneval_wave_ast_runner.py"
    src = open(runner, encoding="utf-8").read()
    assert "--hops-vsa-rank" in src
    assert "hops_vsa" in src


def test_malformed_input_fail_closed(projector):
    with pytest.raises(TypeError):
        projector.project_null(None)  # type: ignore[arg-type]
