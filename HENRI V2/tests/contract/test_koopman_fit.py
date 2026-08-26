"""Contract tests: K2 reduced-rank action-conditioned Koopman fit."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from koopman_fit import (  # noqa: E402
    FLAG,
    KoopmanFitDisabledError,
    evaluate,
    fit_operator,
    spectral_norm,
)


class _R:
    def __init__(self, ep, step, aid, s, a, n):
        self.episode = ep
        self.step = step
        self.action_id = aid
        self.state_wave = s
        self.action_wave = a
        self.next_wave = n


def _lin_dynamics(seed=0, n_ep=8, steps=6, d=8, n_actions=2):
    """Linear per-action dynamics: next = M_a s + noise; dictionary = cat(s, a).

    M_a is a genuine rank-2 contractive map (singular values 0.9, 0.4) so a
    low-rank operator can recover it and open-loop rollouts track. n_ep=8
    gives 24 calibration records (12 per action) — enough to identify the
    low-rank structure under ridge without being degenerate.
    """
    g = np.random.default_rng(seed)
    ms = {}
    for i in range(n_actions):
        U, _ = np.linalg.qr(g.standard_normal((d, 2)))
        V, _ = np.linalg.qr(g.standard_normal((d, 2)))
        ms[f"a{i}"] = U @ np.diag([0.9, 0.4]) @ V.T
    records = []
    for ep in range(n_ep):
        s = g.standard_normal(d)
        s = s / np.linalg.norm(s)
        for st in range(steps):
            a = f"a{st % n_actions}"
            a_wave = np.zeros((1, 8), dtype=np.float32)
            a_wave[0, st % n_actions] = 1.0
            s_t = torch.tensor(s.reshape(1, 8), dtype=torch.float32)
            a_w = torch.tensor(a_wave, dtype=torch.float32)
            nxt = ms[a] @ s + 1e-5 * g.standard_normal(d)
            nxt = nxt / np.linalg.norm(nxt)
            n_w = torch.tensor(nxt.reshape(1, 8), dtype=torch.float32)
            records.append(_R(f"e{ep}", st, a, s_t, a_w, n_w))
            s = nxt
    return records


def _dict_fn(s, a):
    return torch.cat([s.reshape(-1), a.reshape(-1)])


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def test_c1_default_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(KoopmanFitDisabledError):
        evaluate([], [], _dict_fn)


def test_c2_fit_operator_recovery(flag_on):
    g = np.random.default_rng(1)
    phi = torch.tensor(g.standard_normal((20, 16)), dtype=torch.float32)
    M = torch.tensor(g.standard_normal((8, 16)), dtype=torch.float32)
    y = phi @ M.T  # [20, 8] exactly rank-8 target
    op = fit_operator(phi, y, ridge=1e-8, r=8)
    pred = op["V"] @ (op["W"].T @ phi.T)
    err = float(((pred.T - y) ** 2).mean().sqrt())
    assert err < 1e-2


def test_c3_spectral_norm_known(flag_on):
    op = {"V": torch.eye(8)[:, :2], "W": torch.eye(16)[:, :2] * 0.7}
    # K = V W^T: top singular value = 0.7 (orthonormal factors)
    assert abs(spectral_norm(op, iters=60) - 0.7) < 1e-2


def test_c4_arms_skill_and_controls(flag_on):
    recs = _lin_dynamics()
    cal, evl = recs[: len(recs) // 2], recs[len(recs) // 2:]
    out = evaluate(cal, evl, _dict_fn, ridge=1e-6, rank=8, num_blocks=1,
                   horizons=(3, 5))
    os_ = out["one_step"]
    assert os_["action_conditioned"] < os_["persistence"]
    assert os_["skill_ratio_vs_persistence"] > 1.0
    assert os_["shuffled_action"] >= os_["action_conditioned"]
    assert out["verdict"] == "KOOPMAN_FIT_SUPPORTED"
    assert out["trainable_parameters"] == 0
    assert out["optimizer"] is None


def test_c5_rollout_open_loop(flag_on):
    recs = _lin_dynamics(n_ep=10, steps=8)
    cal, evl = recs[: len(recs) // 2], recs[len(recs) // 2:]
    out = evaluate(cal, evl, _dict_fn, ridge=1e-6, rank=8, num_blocks=1,
                   horizons=(3, 5))
    r3 = out["rollouts"]["3"]
    assert r3["conditioned"] < r3["persistence"]

def test_c6_engagement(flag_on):
    recs = _lin_dynamics()
    cal, evl = recs[: len(recs) // 2], recs[len(recs) // 2:]
    out = evaluate(cal, evl, _dict_fn, ridge=1e-6, rank=8, num_blocks=1)
    assert out["engagement"]["cal_pred_cos_conditioned"] > \
        out["engagement"]["cal_pred_cos_persistence"] + 1e-6
