"""Contract tests for the G2 lexical snapping carrier (default-OFF sidecar).

Mechanics-only on CPU with toy per-block unit waves. The LIVE G2 run uses the
real Channel-T codec frame on Vast CUDA with n=24, seed 20260825, tau=0.125.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

try:
    from universal_wave_harness.lexical_snap import (
        DEFAULT_TAU, hopfield_energy, is_enabled, memory_sha256,
        pre_snap_stats, scores_for,
    )
except ImportError:  # pragma: no cover - suite context without package root
    pytest.skip("package root not importable", allow_module_level=True)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _unit_waves(n=6, blocks=32, seed=11, dim=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, blocks, dim, generator=g, dtype=torch.float64)
    return x / x.norm(dim=-1, keepdim=True)


def _ortho(seed=5, dim=8, blocks=32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(blocks, dim, dim, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(q)
    return q


def _run_module_main(env_extra):
    env = dict(os.environ)
    env.pop("HENRI_LEXICAL_SNAP", None)
    env.update(env_extra)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(
        [sys.executable, "-m", "universal_wave_harness.lexical_snap",
         "--csv", "nope.csv", "--n", "2"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    return p


def test_default_off_refuses_without_flag():
    p = _run_module_main({})
    assert p.returncode == 0
    assert "BLOCKED_DEFAULT_OFF" in p.stdout


def test_flag_enables_but_missing_csv_fails_closed():
    p = _run_module_main({"HENRI_LEXICAL_SNAP": "1"})
    # Env gate passed; execution then fails closed on the missing CSV.
    assert "BLOCKED_DEFAULT_OFF" not in p.stdout


def test_frozen_state_no_trainable_params():
    from universal_wave_harness import lexical_snap as m
    src = textwrap.dedent(inspect_getsource(m))
    assert "nn.Parameter(" not in src
    assert "torch.optim" not in src
    assert ".backward(" not in src


def inspect_getsource(mod):
    import inspect
    return inspect.getsource(mod)


def test_memory_hash_deterministic_and_frozen():
    keys = _unit_waves()
    h1 = memory_sha256(keys, "frame1", "enc1")
    keys2 = keys.clone()
    h2 = memory_sha256(keys2, "frame1", "enc1")
    assert h1 == h2
    h3 = memory_sha256(keys, "frame1", "enc2")
    assert h1 != h3  # encoder manifest participates


def test_identity_and_repeat_byte_identical():
    q = _unit_waves(n=4, blocks=32, seed=3)
    keys = _unit_waves(n=4, blocks=32, seed=7)
    s0 = [pre_snap_stats(scores_for(q[i], keys), DEFAULT_TAU) for i in range(4)]
    s1 = [pre_snap_stats(scores_for(q[i], keys), DEFAULT_TAU) for i in range(4)]
    assert json.dumps(s0, sort_keys=True) == json.dumps(s1, sort_keys=True)


def test_joint_spin3_pre_snap_invariance():
    from universal_wave_harness.relational_egress import rotor_sandwich_matrix
    q = _unit_waves(n=4, blocks=32, seed=3)
    keys = _unit_waves(n=4, blocks=32, seed=7)
    T = torch.stack([rotor_sandwich_matrix(1.2, 4)] * 32)
    qT = torch.einsum("kab,nkb->nka", T, q)
    kT = torch.einsum("kab,nkb->nka", T, keys)
    err = 0.0
    for i in range(4):
        a = pre_snap_stats(scores_for(q[i], keys), DEFAULT_TAU)
        b = pre_snap_stats(scores_for(qT[i], kT), DEFAULT_TAU)
        err = max(err, max(abs(x - y) for x, y in zip(a["scores"], b["scores"])))
        assert a["snapped_id"] == b["snapped_id"]
    assert err < 1e-9


def test_query_only_sensitivity():
    from universal_wave_harness.relational_egress import rotor_sandwich_matrix
    q = _unit_waves(n=4, blocks=32, seed=3)
    keys = _unit_waves(n=4, blocks=32, seed=7)
    T = torch.stack([rotor_sandwich_matrix(1.5, 5)] * 32)
    qT = torch.einsum("kab,nkb->nka", T, q)
    delta = 0.0
    for i in range(4):
        a = pre_snap_stats(scores_for(q[i], keys), DEFAULT_TAU)
        b = pre_snap_stats(scores_for(qT[i], keys), DEFAULT_TAU)
        delta = max(delta, max(abs(x - y) for x, y in zip(a["scores"], b["scores"])))
    assert delta > 1e-2


def test_mismatched_frame_collapse():
    q = _unit_waves(n=6, blocks=32, seed=3)
    keys = _unit_waves(n=6, blocks=32, seed=7)
    O = _ortho(seed=9, blocks=32)
    kD = torch.einsum("kab,nkb->nka", O, keys)
    hits = sum(1 for i in range(6)
               if pre_snap_stats(scores_for(q[i], kD), DEFAULT_TAU)["snapped_id"] == i)
    assert hits / 6 <= 0.5


def test_dead_memory_not_confident():
    dead = torch.zeros(6, dtype=torch.float64)
    st = pre_snap_stats(dead, DEFAULT_TAU)
    assert st["p_top1"] <= 0.5
    assert st["entropy_nats"] > 1.0


def test_empty_memory_fails_closed():
    with pytest.raises(ValueError, match="empty score vector"):
        pre_snap_stats(torch.tensor([], dtype=torch.float64), DEFAULT_TAU)


def test_nonfinite_fails_closed():
    with pytest.raises(ValueError, match="non-finite"):
        pre_snap_stats(torch.tensor([1.0, float("nan"), 0.5]), DEFAULT_TAU)


def test_pre_snap_telemetry_before_snap_fields_present():
    q = _unit_waves(n=1, blocks=32, seed=3)[0]
    keys = _unit_waves(n=4, blocks=32, seed=7)
    st = pre_snap_stats(scores_for(q, keys), DEFAULT_TAU)
    for k in ("hopfield_energy", "entropy_nats", "p_top1", "p_top1_idx",
              "p_margin", "s_margin", "participation_ratio", "top5_scores",
              "scores", "snapped_id", "tau", "beta"):
        assert k in st
    assert st["p_top1_idx"] == st["snapped_id"]


def test_hopfield_energy_matches_reference_formula():
    s = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    e = hopfield_energy(s, 0.125)
    ref = -0.125 * torch.logsumexp(s / 0.125, dim=0).item()
    assert abs(e - ref) < 1e-12


def test_no_vacuous_all():
    s = torch.zeros(2, dtype=torch.float64)
    st = pre_snap_stats(s, DEFAULT_TAU)
    assert st["participation_ratio"] > 1.0  # non-degenerate computation
