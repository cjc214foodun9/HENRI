"""Contract tests: K1 Koopman identifiability audit (default-OFF)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from koopman_identifiability import (  # noqa: E402
    FLAG,
    IdentifiabilityDisabledError,
    TransitionRecord,
    audit,
    split_episodes,
)


def _rec(ep, step, aid, seed, digest_prefix=""):
    g = np.random.default_rng(seed)
    w = g.standard_normal((2, 8))
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    return TransitionRecord(ep, step, aid, w.astype(np.float32),
                            w.astype(np.float32), w.astype(np.float32),
                            f"{digest_prefix}{ep}:{step}:t",
                            f"{digest_prefix}{ep}:{step}:n")


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def test_c1_default_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(IdentifiabilityDisabledError):
        audit([], [], [1, 2])


def test_c2_pass_rank_and_math(flag_on):
    # 3 actions x 8 rows: N_a=8 -> r=2 passes (8 >= 8), r=3 blocked (8 < 12)
    recs = [_rec(f"e{i // 8}", i % 8, f"a{i // 8}", i)
            for i in range(24)]
    cal, evl, _, _ = split_episodes(recs, seed=0, eval_frac=0.0)
    out = audit(cal, evl, [1, 2, 3])
    assert out["verdict"] == "IDENTIFIABILITY_PASS(r=1)"
    assert out["recommended_rank"] == 1
    assert out["N_cal"] == 24
    assert out["algebraic_rank_ceiling"] == min(24, 16)
    assert out["per_action"]["a0"]["n"] == 8
    assert out["rank_checks"][1]["pass"] is True
    assert out["rank_checks"][3]["pass"] is False


def test_c3_blocked_on_small_support(flag_on):
    # 3 actions x 2 rows: N_a=2 < 4r for every r >= 1
    recs = [_rec(f"e{i // 2}", i % 2, f"a{i // 2}", i) for i in range(6)]
    cal, evl, _, _ = split_episodes(recs, seed=0, eval_frac=0.0)
    out = audit(cal, evl, [1, 2])
    assert out["verdict"] == "IDENTIFIABILITY_BLOCKED"
    assert out["recommended_rank"] is None


def test_c4_episode_split_disjoint_and_exhaustive(flag_on):
    recs = [_rec(f"e{i % 5}", i // 5, f"a{i % 2}", i) for i in range(50)]
    cal, evl, cal_ids, eval_ids = split_episodes(recs, seed=7, eval_frac=0.3)
    assert set(cal_ids) & set(eval_ids) == set()
    assert set(cal_ids) | set(eval_ids) == {f"e{i}" for i in range(5)}
    assert all(r.episode in cal_ids for r in cal)
    assert all(r.episode in eval_ids for r in evl)


def test_c5_participation_ratio_peaked_spectrum(flag_on):
    # X with exact singular values [4,4,4,4] -> PR = (64)^2 / (4*256) = 4.0
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((24, 4)))
    V, _ = np.linalg.qr(rng.standard_normal((16, 4)))
    X = (U * np.array([4.0, 4.0, 4.0, 4.0])) @ V.T  # [24, 16]
    recs = [TransitionRecord(f"e{i % 3}", i, f"a{i % 2}", X[i].reshape(2, 8).astype(np.float32),
                             X[i].reshape(2, 8).astype(np.float32),
                             X[i].reshape(2, 8).astype(np.float32), f"t{i}", f"n{i}")
            for i in range(24)]
    cal, evl, _, _ = split_episodes(recs, seed=0, eval_frac=0.0)
    out = audit(cal, evl, [2])
    assert abs(out["participation_ratio"] - 4.0) < 0.1
    assert out["numerical_rank"] == 4


def test_c6_eval_overlap_reported(flag_on):
    recs = [_rec(f"e{i // 8}", i % 8, f"a{i // 8}", i) for i in range(24)]
    cal, evl, _, _ = split_episodes(recs, seed=3, eval_frac=0.25)
    out = audit(cal, evl, [1])
    assert 0.0 <= out["eval_overlap"] <= 1.0
