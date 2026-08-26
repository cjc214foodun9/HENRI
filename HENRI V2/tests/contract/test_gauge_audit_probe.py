"""Contract tests for the G0 gauge-audit probe (default-OFF sidecar).

Mechanics-only on CPU: Cl(3,0) multiplication table, rotor sandwich
orthogonality/grade-preservation, grade-scramble discriminates O(8) vs
Spin(3), relational joint-stability, invariant-collision flag, default-OFF.
The LIVE gauge audit loads the real Channel-T waves + checkpoint on Vast.
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from universal_wave_harness.gauge_audit import (  # noqa: E402
    TABLE,
    grade_projectors,
    grade_scramble,
    left_mult_matrix,
    random_orthogonal,
    reversion_matrix,
    right_mult_matrix,
    rotor_sandwich,
)

PROBE = os.path.join(REPO, "universal_wave_harness", "gauge_audit.py")


def _toy_waves(num_blocks=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_blocks, 8, generator=g, dtype=torch.float64)
    return x / x.norm(dim=-1, keepdim=True)


def test_cl30_multiplication_basics():
    # e_i^2 = +1 (scalar, +1)
    for i in (1, 2, 3):
        k, s = TABLE[(i, i)]
        assert k == 0 and s == 1
    # bivectors and pseudoscalar square to -1
    for b in (4, 5, 6, 7):
        k, s = TABLE[(b, b)]
        assert k == 0 and s == -1
    # e1*e2 = e12 ; e2*e1 = -e12 ; e1*e23 = e123 ; e1*e31 = -e3
    assert TABLE[(1, 2)] == (4, 1)
    assert TABLE[(2, 1)] == (4, -1)
    assert TABLE[(1, 5)] == (7, 1)
    assert TABLE[(1, 6)] == (3, -1)


def test_reversion_matrix():
    rv = reversion_matrix()
    for i in range(8):
        expected = -1.0 if i in (4, 5, 6, 7) else 1.0
        assert rv[i, i].item() == expected


def test_rotor_sandwich_is_orthogonal():
    R = rotor_sandwich(1.2, 4)
    err = float((R.T @ R - torch.eye(8, dtype=torch.float64)).abs().max())
    assert err < 1e-12


def test_rotor_preserves_grade_energy():
    R = rotor_sandwich(0.9, 6)
    P = grade_projectors()
    x = torch.randn(8, dtype=torch.float64)
    Rx = R @ x
    for gr in range(4):
        before = float((P[gr] @ x).pow(2).sum())
        after = float((P[gr] @ Rx).pow(2).sum())
        assert abs(before - after) < 1e-9


def test_grade_scramble_discriminates_groups():
    P = grade_projectors()
    ident = torch.eye(8, dtype=torch.float64)
    assert grade_scramble(ident, P) < 1e-12
    spin3 = rotor_sandwich(1.3, 5)
    assert grade_scramble(spin3, P) < 1e-6
    o8 = random_orthogonal(seed=99, dim=8, count=1)[0]
    assert grade_scramble(o8, P) > 0.2


def test_relational_joint_stability_vs_xonly():
    x = _toy_waves()
    k = _toy_waves(seed=8)
    T = torch.stack([rotor_sandwich(0.5, 4)] * 4)
    s = float(torch.cosine_similarity(x.reshape(-1), k.reshape(-1), dim=0))
    Tx = torch.einsum("kab,kb->ka", T, x)
    Tk = torch.einsum("kab,kb->ka", T, k)
    s_joint = float(torch.cosine_similarity(Tx.reshape(-1), Tk.reshape(-1), dim=0))
    assert abs(s - s_joint) < 1e-9
    s_xonly = float(torch.cosine_similarity(Tx.reshape(-1), k.reshape(-1), dim=0))
    assert abs(s - s_xonly) > 1e-2


def test_invariant_collision_flag():
    # unit-row waves: per-block norms are the only O(8)^K invariants -> all 1
    x = _toy_waves(num_blocks=8, seed=3)
    y = _toy_waves(num_blocks=8, seed=4)
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    assert torch.allclose(nx, ny, atol=1e-6)


def test_default_off_refuses_without_flag():
    env = dict(os.environ)
    env.pop("HENRI_GAUGE_AUDIT", None)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, PROBE, "--csv", "x.csv"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert "BLOCKED_DEFAULT_OFF" in r.stdout
    assert r.returncode == 0
