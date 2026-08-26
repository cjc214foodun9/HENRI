"""Contract tests for the G1 relational egress kernel (default-OFF sidecar).

Mechanics-only on CPU with toy per-block unit waves. The LIVE G1 run uses the
real Channel-T codec frame on Vast.
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from universal_wave_harness.relational_egress import (  # noqa: E402
    per_block_cosine_mean,
    rotor_sandwich_matrix,
    score_all,
)
from universal_wave_harness.gauge_audit import random_orthogonal  # noqa: E402

PROBE = os.path.join(REPO, "universal_wave_harness", "relational_egress.py")


def _unit_waves(n=6, blocks=64, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, blocks, 8, generator=g, dtype=torch.float64)
    return x / x.norm(dim=-1, keepdim=True)


def _apply_blockwise(T, x):
    """T [blocks,8,8] x [n,blocks,8] -> [n,blocks,8] per-block."""
    return torch.einsum("kab,nkb->nka", T, x)


def test_per_block_cosine_self_is_one():
    q = _unit_waves(n=2, seed=1)
    assert abs(per_block_cosine_mean(q[0], q[0]) - 1.0) < 1e-9


def test_per_block_cosine_mean_distinct():
    q = _unit_waves(n=2, seed=2)
    s = per_block_cosine_mean(q[0], q[1])
    assert abs(s) < 0.9


def test_joint_spin3_invariance():
    q = _unit_waves(n=2, blocks=16, seed=3)
    T = torch.stack([rotor_sandwich_matrix(1.2, 4)] * 16)
    qT = _apply_blockwise(T, q)
    s0 = per_block_cosine_mean(q[0], q[1])
    s1 = per_block_cosine_mean(qT[0], qT[1])
    assert abs(s0 - s1) < 1e-9


def test_x_only_sensitivity():
    q = _unit_waves(n=2, blocks=128, seed=4)
    T = torch.stack([rotor_sandwich_matrix(2.0, 4)] * 128)
    qx = _apply_blockwise(T, q[:1])[0]
    s0 = per_block_cosine_mean(q[0], q[1])
    s1 = per_block_cosine_mean(qx, q[1])
    assert abs(s0 - s1) > 1e-2


def test_self_hit_rank():
    keys = _unit_waves(n=4, blocks=32, seed=5)
    s = score_all(keys[2], keys)
    assert int(s.argmax().item()) == 2


def test_mismatched_keys_reduce_discrimination():
    # Misaligned frame: per-block ARBITRARY O(8) on keys only (NOT a valid
    # joint Cl(3,0) gauge — G0: grade_scramble ~5600) destroys frame
    # alignment, so self-hit discrimination collapses to chance. A uniform
    # frame rotation is gauge-safe and preserves ranking (valid invariance,
    # not a mismatch control); bounded-angle per-block Spin(3) rotors leave
    # expected self-cos ~0.64 and also do NOT collapse discrimination.
    keys = _unit_waves(n=4, blocks=32, seed=6)
    T = random_orthogonal(seed=99, dim=8, count=32)
    k_mis = _apply_blockwise(T, keys)
    hits = 0
    for i in range(4):
        s = score_all(keys[i], k_mis)
        if int(s.argmax().item()) == i:
            hits += 1
    assert hits / 4 < 0.5


def test_default_off_refuses_without_flag():
    env = dict(os.environ)
    env.pop("HENRI_RELATIONAL_EGRESS", None)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, PROBE, "--csv", "x.csv"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert "BLOCKED_DEFAULT_OFF" in r.stdout
    assert r.returncode == 0
